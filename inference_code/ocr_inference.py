# =========================
# Imports + Environment Setup
# =========================
import os
import re
import time
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime

import pypdfium2 as pdfium
from PIL import Image

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode

# =========================
# DEBUG MODE - SET TO True
# =========================
DEBUG_MODE = True  # ← SET THIS TO True TO ENABLE DEBUGGING
ENABLE_270_ONLY_ROTATION=False

ENABLE_ALL_ROTATIONS=True

USE_BALANCE_ID_STRATEGY=True

TARGET_ALL_PAGES=True
# =========================
# Environment Configuration
# =========================
def str_to_bool(s: Optional[str], default: bool = False) -> bool:
    """Convert string to boolean"""
    if s is None:
        return default
    s = str(s).strip().lower()
    return s in {"1", "true", "yes", "on"}

def load_env_flags() -> Dict[str, bool]:
    """Load environment configuration flags"""
    return {
        "enable_270_only": str_to_bool(ENABLE_270_ONLY_ROTATION, False),
        "enable_all_rotations": str_to_bool(ENABLE_ALL_ROTATIONS, False),
        "use_balance_id_strategy": str_to_bool(USE_BALANCE_ID_STRATEGY, False),
        "target_all_pages": str_to_bool(TARGET_ALL_PAGES, False),
    }

# =========================
# Time Profiling Class
# =========================
class TimeProfiler:
    def __init__(self):
        self.timings = {}
        self.nested = set()

    def time_block(self, name: str, is_nested: bool = False):
        class TimerContext:
            def __init__(self, profiler, label, nested):
                self.profiler = profiler
                self.label = label
                self.nested = nested
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.profiler.timings[self.label] = elapsed
                if self.nested:
                    self.profiler.nested.add(self.label)
                print(f"{'  ' if self.nested else ''}⏱️  {self.label}: {elapsed:.3f}s")

        return TimerContext(self, name, is_nested)

    def print_summary(self):
        print("\n" + "="*60)
        print("TIME PROFILING SUMMARY")
        print("="*60)

        top_level = {k: v for k, v in self.timings.items() if k not in self.nested}
        nested_timings = {k: v for k, v in self.timings.items() if k in self.nested}

        top_level_total = sum(top_level.values())
        nested_total = sum(nested_timings.values())

        print(f"\n{'Top-Level Operations':>40}")
        print("-"*60)
        for name, duration in sorted(top_level.items(), key=lambda x: x[1], reverse=True):
            pct = (duration / top_level_total * 100) if top_level_total > 0 else 0
            print(f"{name:40s}: {duration:7.3f}s ({pct:5.1f}%)")

        if nested_timings:
            print(f"\n{'Nested Operations (within Batch OCR)':>40}")
            print("-"*60)
            for name, duration in sorted(nested_timings.items(), key=lambda x: x[1], reverse=True):
                pct = (duration / nested_total * 100) if nested_total > 0 else 0
                print(f"  {name:38s}: {duration:7.3f}s ({pct:5.1f}%)")

        print("="*60)
        print(f"{'TOTAL (Top-Level Only)':40s}: {top_level_total:7.3f}s")
        print("="*60)

# =========================
# Image Processing Functions
# =========================
def get_page_images(pdf_bytes: bytes, dpi: int = 300, max_pages: Optional[int] = None) -> List[Image.Image]:
    """Extract pages as PIL Images"""
    pdf = pdfium.PdfDocument(BytesIO(pdf_bytes))
    scale = dpi / 72

    num_pages = len(pdf) if max_pages is None else min(max_pages, len(pdf))
    imgs = []
    for i in range(num_pages):
        bitmap = pdf[i].render(scale=scale)
        imgs.append(bitmap.to_pil())

    return imgs

def merge_rotated_pages(rotated_pages: List[Image.Image], angle: int) -> Image.Image:
    """Merge multiple rotated pages based on rotation angle"""
    if len(rotated_pages) == 1:
        return rotated_pages[0]

    if angle == 270:
        total_width = sum(img.width for img in rotated_pages)
        max_height = max(img.height for img in rotated_pages)
        merged = Image.new("RGB", (total_width, max_height), "white")

        x_offset = 0
        for img in rotated_pages:
            merged.paste(img, (x_offset, 0))
            x_offset += img.width
        return merged

    elif angle == 90:
        total_width = sum(img.width for img in rotated_pages)
        max_height = max(img.height for img in rotated_pages)
        merged = Image.new("RGB", (total_width, max_height), "white")

        x_offset = 0
        for img in reversed(rotated_pages):
            merged.paste(img, (x_offset, 0))
            x_offset += img.width
        return merged

    elif angle == 180:
        total_height = sum(img.height for img in rotated_pages)
        max_width = max(img.width for img in rotated_pages)
        merged = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        for img in reversed(rotated_pages):
            merged.paste(img, (0, y_offset))
            y_offset += img.height
        return merged

    else:
        total_height = sum(img.height for img in rotated_pages)
        max_width = max(img.width for img in rotated_pages)
        merged = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        for img in rotated_pages:
            merged.paste(img, (0, y_offset))
            y_offset += img.height
        return merged

def image_to_pdf_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PDF bytes"""
    buf = BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()

# =========================
# Batch OCR Processing
# =========================
def batch_ocr_process(
    pdf_bytes_list: List[bytes],
    lang_list: List[str],
    profiler: TimeProfiler
) -> List[str]:
    """Process multiple PDFs in a single batch call"""

    with profiler.time_block("1. PDF Preprocessing", is_nested=True):
        processed_bytes = []
        for pdf_bytes in pdf_bytes_list:
            processed = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes,
                start_page_id=0,
                end_page_id=None
            )
            processed_bytes.append(processed)

    with profiler.time_block("2. Pipeline Analysis (OCR)", is_nested=True):
        infer_results, all_image_lists, all_pdf_docs, result_lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                processed_bytes,
                lang_list,
                parse_method="ocr",
                formula_enable=False,
                table_enable=True,
            )
        )

    with profiler.time_block("3. JSON Conversion", is_nested=True):
        middle_jsons = []
        for i in range(len(infer_results)):
            middle_json = result_to_middle_json(
                infer_results[i],
                all_image_lists[i],
                all_pdf_docs[i],
                image_writer=None,
                lang=result_lang_list[i],
                ocr_enable=ocr_enabled_list[i],
            )
            middle_jsons.append(middle_json)

    with profiler.time_block("4. Markdown Generation", is_nested=True):
        results = []
        for middle_json in middle_jsons:
            md_text = union_make(middle_json["pdf_info"], MakeMode.MM_MD, "")
            results.append(md_text)

    return results

# =========================
# Validation Logic - WITH DEBUG MODE
# =========================

DEFAULT_RANGE_MIN = Decimal("1.0")
DEFAULT_RANGE_MAX = Decimal("2.0")

def normalize_text(md: str) -> str:
    """Normalize text by removing HTML and extra whitespace"""
    md = re.sub(r"<[^>]+>", " ", md)
    md = re.sub(r"\s+", " ", md)
    return md.strip()

# ===== SOURCE PDF EXTRACTION =====
def extract_instrument_id_doc1(text: str) -> Optional[str]:
    """Extract Instrument ID from source PDF"""
    match = re.search(
        r"Instrument\s*\(Instrument\s*ID\)\s*[:\-]?\s*[^(]*\(\s*([A-Z0-9\-]+)\s*\)",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else None

def extract_target_weight_doc1(text: str) -> Optional[Decimal]:
    """Extract target weight from source PDF"""
    match = re.search(
        r"about\s+([\d]+\.[\d]+)\s*\([^)]+\)\s*g",
        text,
        re.IGNORECASE,
    )
    if match:
        return Decimal(match.group(1))

    match = re.search(r"Maximum\s+([\d]+\.[\d]+)", text, re.IGNORECASE)
    if match:
        val = Decimal(match.group(1))
        if val > Decimal("0.01"):
            return val

    return None

def extract_target_range_doc1(text: str) -> Optional[Tuple[Decimal, Decimal]]:
    """Extract weight range from source PDF"""
    match = re.search(
        r"about\s+[\d]+\.[\d]+\s*\(\s*([\d\.]+)\s+to\s+([\d\.]+)\s*\)",
        text,
        re.IGNORECASE,
    )
    if match:
        return (Decimal(match.group(1)), Decimal(match.group(2)))

    match = re.search(r"\(\s*([\d\.]+)\s*-\s*([\d\.]+)\s*\)", text)
    if match:
        return (Decimal(match.group(1)), Decimal(match.group(2)))

    return None

def extract_used_from_doc1(text: str) -> Optional[str]:
    """Extract 'Used From' datetime from source PDF"""
    match = re.search(
        r"Used\s+From\s+([\d]{2}-[A-Za-z]{3}-[\d]{4}\s+[\d]{2}:[\d]{2})",
        text,
        re.IGNORECASE
    )
    return match.group(1) if match else None

def extract_used_upto_doc1(text: str) -> Optional[str]:
    """Extract 'Used Up to' datetime from source PDF"""
    match = re.search(
        r"Used\s+Up\s+to\s+([\d]{2}-[A-Za-z]{3}-[\d]{4}\s+[\d]{2}:[\d]{2})",
        text,
        re.IGNORECASE
    )
    return match.group(1) if match else None

# ===== TARGET PDF EXTRACTION - WITH DEBUGGING =====
def split_weighing_blocks(doc2: str) -> List[str]:
    """Split document by WEIGHING PROTOCOL with debug output"""
    pattern = re.compile(r'WEIGHING\s+PROTOCOL', re.IGNORECASE)
    matches = list(pattern.finditer(doc2))

    if DEBUG_MODE:
        print("\n" + "="*80)
        print("DEBUG: split_weighing_blocks()")
        print("="*80)
        print(f"Found {len(matches)} 'WEIGHING PROTOCOL' occurrences")

    if not matches:
        if DEBUG_MODE:
            print("⚠️  No 'WEIGHING PROTOCOL' found in document!")
        return []

    blocks = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(doc2)

        block_text = doc2[start_pos:end_pos].strip()
        if block_text:
            blocks.append(block_text)

            if DEBUG_MODE:
                print(f"\nBlock {i+1}:")
                print(f"  Position: {start_pos} to {end_pos} ({len(block_text)} chars)")
                print(f"  First 300 chars: {block_text[:300]}")
                print(f"  Last 200 chars: ...{block_text[-200:]}")

    return blocks

def extract_balance_id_from_block(block: str) -> Optional[str]:
    """Extract Balance ID with debug output"""
    match = re.search(
        r'Balance\s*ID\s*[:\-]?\s*([A-Z0-9\-]+)',
        block,
        re.IGNORECASE
    )
    if match:
        if DEBUG_MODE:
            print(f"    ✅ Balance ID found (Pattern 1): {match.group(1)}")
        return match.group(1)

    match = re.search(
        r'(?<!Instrument\s)ID\s*[:\-]?\s*([A-Z0-9\-]+)',
        block,
        re.IGNORECASE
    )
    if match:
        if DEBUG_MODE:
            print(f"    ✅ Balance ID found (Pattern 2): {match.group(1)}")
        return match.group(1)

    if DEBUG_MODE:
        print("    ❌ Balance ID not found")
    return None

def extract_maximum_from_block(block: str) -> Optional[Decimal]:
    """Extract Maximum weight with debug output"""
    match = re.search(
        r'Maximum\s+([\d]+\.?[\d]*)\s*g?',
        block,
        re.IGNORECASE
    )
    if match:
        if DEBUG_MODE:
            print(f"    ✅ Maximum found (Pattern 1): {match.group(1)}")
        return Decimal(match.group(1))

    match = re.search(
        r'Max\s*:?\s+([\d]+\.?[\d]*)',
        block,
        re.IGNORECASE
    )
    if match:
        if DEBUG_MODE:
            print(f"    ✅ Maximum found (Pattern 2): {match.group(1)}")
        return Decimal(match.group(1))

    if DEBUG_MODE:
        print("    ❌ Maximum not found")
    return None

def extract_datetime_from_block(block: str) -> Optional[str]:
    """
    Extract datetime with comprehensive fallback patterns.
    Handles both formats:
      - "07.12.2025 11:18" (with space)
      - "07.12.202511:19" (without space)
    
    Automatically normalizes output to include space between date and time.
    """
    
    if DEBUG_MODE:
        print("\n    🔍 DEBUG: Searching for datetime...")
        print(f"    Block length: {len(block)} chars")
        print(f"    First 400 chars of block:")
        print(f"    {repr(block[:400])}")
        print()

    # Pattern 1-3: Strict - RIGHT after WEIGHING PROTOCOL
    # \s* makes space OPTIONAL between date and time
    for sep_name, sep in [('dots', '\\.'), ('dashes', '\\-'), ('slashes', '/')]:
        pattern = f'WEIGHING\\s+PROTOCOL\\s*\\n?\\s*([\\d]{{2}}{sep}[\\d]{{2}}{sep}[\\d]{{4}}\\s*[\\d]{{2}}:[\\d]{{2}})'
        match = re.search(pattern, block, re.IGNORECASE)
        if match:
            result = match.group(1)
            # Normalize: Add space if missing between date and time
            if ' ' not in result:
                # Split at position 10: "07.12.202511:19" -> "07.12.2025 11:19"
                result = result[:10] + ' ' + result[10:]
            if DEBUG_MODE:
                print(f"    ✅ Pattern 1-3 ({sep_name}): Found '{result}'")
            return result
        elif DEBUG_MODE:
            print(f"    ❌ Pattern 1-3 ({sep_name}): Not found")

    # Pattern 4: Search first 10 lines
    if DEBUG_MODE:
        print("\n    Trying Pattern 4: First 10 lines")
    
    lines = block.split('\n')[:10]
    if DEBUG_MODE:
        print(f"    Total lines to check: {len(lines)}")
        for idx, line in enumerate(lines):
            print(f"    Line {idx}: {repr(line[:100])}")
    
    for idx, line in enumerate(lines):
        # \s* makes space optional between date and time
        match = re.search(r'([\d]{2}[.\-/][\d]{2}[.\-/][\d]{4}\s*[\d]{2}:[\d]{2})', line)
        if match:
            result = match.group(1)
            # Normalize: Add space if missing
            if ' ' not in result:
                result = result[:10] + ' ' + result[10:]
            if DEBUG_MODE:
                print(f"    ✅ Pattern 4: Found '{result}' in line {idx}")
            return result

    if DEBUG_MODE:
        print("    ❌ Pattern 4: Not found in first 10 lines")

    # Pattern 5: Last resort - anywhere in block
    if DEBUG_MODE:
        print("\n    Trying Pattern 5: Anywhere in block")
    
    match = re.search(r'([\d]{2}[.\-/][\d]{2}[.\-/][\d]{4}\s*[\d]{2}:[\d]{2})', block)
    if match:
        result = match.group(1)
        # Normalize: Add space if missing
        if ' ' not in result:
            result = result[:10] + ' ' + result[10:]
        if DEBUG_MODE:
            print(f"    ✅ Pattern 5: Found '{result}'")
        return result

    if DEBUG_MODE:
        print("    ❌ Pattern 5: Not found anywhere")
        print("\n    ⚠️  NO DATETIME FOUND IN THIS BLOCK!")
        print("    This block will have 'weighing_datetime': None")

    return None


# ===== VALIDATION HELPERS =====
def parse_datetime_source(dt_str: str) -> Optional[datetime]:
    """Parse source datetime format: 07-Dec-2025 11:16"""
    try:
        return datetime.strptime(dt_str, "%d-%b-%Y %H:%M")
    except:
        return None

def parse_datetime_target(dt_str: str) -> Optional[datetime]:
    """Parse target datetime format with multiple fallbacks"""
    for fmt in ["%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M"]:
        try:
            return datetime.strptime(dt_str, fmt)
        except:
            continue
    return None

def validate_weight(target: Decimal, measured: Decimal, tolerance: Decimal = Decimal("0.0001")) -> bool:
    """Check if measured weight matches target within tolerance"""
    return abs(target - measured) <= tolerance

def validate_datetime_in_range(dt: datetime, start: datetime, end: datetime) -> bool:
    """Check if datetime is within range (inclusive)"""
    return start <= dt <= end

def count_balance_ids_in_text(text: str) -> int:
    """Count number of Balance ID occurrences in OCR text"""
    blocks = split_weighing_blocks(text)
    count = 0
    for block in blocks:
        if extract_balance_id_from_block(block):
            count += 1
    return count

def validate_document_pair(doc1_text: str, doc2_text: str) -> Dict:
    """Main validation function with debug output - ALWAYS returns full structure"""
    d1 = normalize_text(doc1_text)

    # Extract source PDF data
    instrument_id = extract_instrument_id_doc1(d1)
    target_weight = extract_target_weight_doc1(d1)
    target_range = extract_target_range_doc1(d1)
    used_from_str = extract_used_from_doc1(d1)
    used_upto_str = extract_used_upto_doc1(d1)

    used_from_dt = parse_datetime_source(used_from_str) if used_from_str else None
    used_upto_dt = parse_datetime_source(used_upto_str) if used_upto_str else None

    if target_range:
        range_min, range_max = target_range
        print(f"✅ Using extracted range from source PDF: {range_min} to {range_max}")
    else:
        range_min, range_max = DEFAULT_RANGE_MIN, DEFAULT_RANGE_MAX
        print(f"⚠️  Range not found in source PDF, using defaults: {range_min} to {range_max}")

    # Build source_ocr_result regardless of whether data is found
    source_ocr_result = {
        "instrument_id": instrument_id if instrument_id else None,
        "target_weight_g": str(target_weight) if target_weight else None,
        "weight_range": {
            "min_g": str(range_min),
            "max_g": str(range_max)
        },
        "used_from": used_from_str,
        "used_upto": used_upto_str
    }

    # If critical source data is missing, return early with full structure
    if not instrument_id or target_weight is None:
        return {
            "source_ocr_result": source_ocr_result,
            "target_ocr_result": {
                "weighing_protocols": []
            },
            "validation": {
                "weight_validated": False,
                "datetime_validated": False,
                "overall_validated": False
            },
            "error": "Instrument ID or target weight missing in source document"
        }

    # Process target PDF
    d2 = normalize_text(doc2_text)
    blocks = split_weighing_blocks(d2)

    if DEBUG_MODE:
        print("\n" + "="*80)
        print("DEBUG: Processing each block")
        print("="*80)

    target_records = []
    datetime_validation_passed = True

    for idx, block in enumerate(blocks):
        if DEBUG_MODE:
            print(f"\n📦 Processing Block {idx+1}/{len(blocks)}")

        balance_id = extract_balance_id_from_block(block)
        if not balance_id:
            if DEBUG_MODE:
                print(f"  ⏭️  Skipping block {idx+1}: No Balance ID found")
            continue

        max_weight = extract_maximum_from_block(block)
        if max_weight is None:
            if DEBUG_MODE:
                print(f"  ⏭️  Skipping block {idx+1}: No Maximum weight found")
            continue

        datetime_str = extract_datetime_from_block(block)
        datetime_dt = parse_datetime_target(datetime_str) if datetime_str else None

        # Weight validation
        balance_matches_instrument = (balance_id == instrument_id)
        in_range = range_min <= max_weight <= range_max
        matches_target = validate_weight(target_weight, max_weight)
        difference = abs(max_weight - target_weight)

        # Datetime validation
        datetime_in_range = None
        if datetime_dt and used_from_dt and used_upto_dt:
            datetime_in_range = validate_datetime_in_range(datetime_dt, used_from_dt, used_upto_dt)
            if balance_matches_instrument and not datetime_in_range:
                datetime_validation_passed = False

        if DEBUG_MODE:
            print(f"\n  📊 Block {idx+1} Results:")
            print(f"    Balance ID: {balance_id} (Matches: {balance_matches_instrument})")
            print(f"    Maximum: {max_weight}")
            print(f"    Datetime: {datetime_str} (Parsed: {datetime_dt})")
            print(f"    In Range: {in_range}, Matches Target: {matches_target}")
            print(f"    Datetime in Used Range: {datetime_in_range}")

        target_records.append({
            "balance_id": balance_id,
            "matches_instrument_id": balance_matches_instrument,
            "weighing_datetime": datetime_str,
            "maximum_g": str(max_weight),
            "weight_in_range": in_range,
            "weight_matches_target": matches_target,
            "weight_difference": str(difference),
            "datetime_in_used_range": datetime_in_range
        })

    target_ocr_result = {
        "weighing_protocols": target_records
    }

    matching_instrument_records = [r for r in target_records if r["matches_instrument_id"]]
    weight_validation_passed = any(
        (r["weight_in_range"] or r["weight_matches_target"]) 
        for r in matching_instrument_records
    ) if matching_instrument_records else False

    overall_validated = weight_validation_passed and datetime_validation_passed

    if DEBUG_MODE:
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Weight Validated: {weight_validation_passed}")
        print(f"Datetime Validated: {datetime_validation_passed}")
        print(f"Overall Validated: {overall_validated}")
        print("="*80)

    # ALWAYS return full structure
    result = {
        "source_ocr_result": source_ocr_result,
        "target_ocr_result": target_ocr_result,
        "validation": {
            "weight_validated": weight_validation_passed,
            "datetime_validated": datetime_validation_passed,
            "overall_validated": overall_validated
        }
    }

    # Add error message if no weighing protocols found
    if not target_records:
        result["error"] = "No weighing protocols found in target document"

    return result


# =========================
# Rotation Strategy Functions
# =========================
def determine_rotation_angles(flags: Dict[str, bool]) -> List[int]:
    """Determine which rotation angles to test"""
    if flags["enable_all_rotations"]:
        return [0, 90, 180, 270]
    elif flags["enable_270_only"]:
        return [270]
    else:
        return [270]

def process_target_with_rotation_strategy(
    target_pdf_bytes: bytes,
    flags: Dict[str, bool],
    profiler: TimeProfiler
) -> Tuple[bytes, str, int]:
    """Process target PDF with rotation strategy"""

    pdf_doc = pdfium.PdfDocument(BytesIO(target_pdf_bytes))
    actual_page_count = len(pdf_doc)
    pdf_doc.close()

    print(f"\n📄 PDF Analysis:")
    print(f"   • Total pages in PDF: {actual_page_count}")

    
    max_pages = None
    pages_to_process = actual_page_count
    print(f"   • Mode: Process ALL {actual_page_count} pages")
    

    with profiler.time_block("B. Extract Target Pages as Images"):
        page_imgs = get_page_images(target_pdf_bytes, dpi=300, max_pages=max_pages)

    print(f"   • Extracted: {len(page_imgs)} image(s)")

    debug_dir = Path("ocr_debug_images")
    debug_dir.mkdir(exist_ok=True)

    print(f"\n💾 Saving original pages to {debug_dir}/")
    for idx, img in enumerate(page_imgs, start=1):
        img_path = debug_dir / f"step1_original_page_{idx}.png"
        img.save(img_path)
        print(f"   • Page {idx}: {img_path.name} ({img.width}x{img.height})")

    angles = determine_rotation_angles(flags)
    print(f"\n📐 Rotation Strategy:")
    print(f"   • Testing angles: {angles}")
    print(f"   • Strategy: {'BALANCE ID counting' if flags['use_balance_id_strategy'] else 'Default (first angle)'}")

    rotation_pdfs = []

    with profiler.time_block("C. Rotate & Merge Pages"):
        for angle in angles:
            print(f"\n   🔄 Processing {angle}° rotation:")

            rotated_pages = []
            for page_idx, img in enumerate(page_imgs, start=1):
                rotated_img = img.rotate(angle, expand=True)
                rotated_pages.append(rotated_img)

                rot_path = debug_dir / f"step2_rotated_{angle}deg_page_{page_idx}.png"
                rotated_img.save(rot_path)
                print(f"      • Page {page_idx}: rotated to {angle}° ({rotated_img.width}x{rotated_img.height})")

            if len(rotated_pages) == 1:
                merged_img = rotated_pages[0]
                print(f"      • Single page - using as-is")
            else:
                merged_img = merge_rotated_pages(rotated_pages, angle)
                print(f"      • Merged {len(rotated_pages)} pages into {merged_img.width}x{merged_img.height}")

            merged_path = debug_dir / f"step3_final_{angle}deg_merged.png"
            merged_img.save(merged_path)
            print(f"      ✅ Final: {merged_path.name}")

            rotated_pdf_bytes = image_to_pdf_bytes(merged_img)
            rotation_pdfs.append((angle, rotated_pdf_bytes))

            pdf_path = debug_dir / f"step4_ocr_input_{angle}deg.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(rotated_pdf_bytes)
            print(f"      💾 PDF: {pdf_path.name} ({len(rotated_pdf_bytes)} bytes)")

    print(f"\n🔍 OCR Processing:")
    print(f"   • Running OCR on {len(rotation_pdfs)} rotation(s)...")

    all_pdf_bytes = [pdf_bytes for _, pdf_bytes in rotation_pdfs]
    all_langs = ["ch"] * len(all_pdf_bytes)

    with profiler.time_block("D. Batch OCR Processing"):
        ocr_results = batch_ocr_process(all_pdf_bytes, all_langs, profiler)

    print(f"\n📝 Saving OCR results:")
    for idx, (angle, _) in enumerate(rotation_pdfs):
        text_path = debug_dir / f"step5_ocr_output_{angle}deg.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"=== OCR OUTPUT FOR {angle}° ROTATION ===\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Character count: {len(ocr_results[idx])}\n")
            f.write("="*60 + "\n\n")
            f.write(ocr_results[idx])

        preview = ocr_results[idx][:300].replace('\n', ' ')
        print(f"   • {angle}° ({len(ocr_results[idx])} chars): {preview}...")

    if flags["use_balance_id_strategy"] and len(angles) > 1:
        print(f"\n⚖️  Applying BALANCE ID Strategy:")

        scores = []
        for idx, (angle, pdf_bytes) in enumerate(rotation_pdfs):
            ocr_text = ocr_results[idx]
            balance_id_count = count_balance_ids_in_text(ocr_text)
            scores.append((balance_id_count, angle, pdf_bytes, ocr_text))
            print(f"   • {angle:3d}°: {balance_id_count} Balance ID(s) detected")

        scores.sort(reverse=True, key=lambda x: x[0])
        best_count, best_angle, best_pdf_bytes, best_ocr_text = scores[0]

        print(f"\n✅ Selected: {best_angle}° rotation ({best_count} Balance IDs)")

        return best_pdf_bytes, best_ocr_text, best_angle

    else:
        best_angle, best_pdf_bytes = rotation_pdfs[0]
        best_ocr_text = ocr_results[0]

        print(f"\n✅ Using default: {best_angle}° rotation")

        return best_pdf_bytes, best_ocr_text, best_angle

# =========================
# Main Processing Function
# =========================
def process_ocr_validation(
    source_pdf_bytes: bytes, 
    target_pdf_bytes: bytes,
    progress_callback=None
) -> Dict:
    """Main entry point for OCR validation processing"""
    profiler = TimeProfiler()
    flags = load_env_flags()

    print("\n" + "="*60)
    print("BATCH OCR PROCESSING WITH VALIDATION")
    print("="*60)

    if DEBUG_MODE:
        print("\n⚠️  DEBUG MODE ENABLED - Verbose output active")

    print("\n🔧 Configuration:")
    print(f"   • 270° only rotation: {flags['enable_270_only']}")
    print(f"   • All rotations: {flags['enable_all_rotations']}")
    print(f"   • BALANCE ID strategy: {flags['use_balance_id_strategy']}")
    print(f"   • Process all target pages: {flags['target_all_pages']}")
    print()

    try:
        # ========================================
        # STAGE 1: Read and preprocess PDFs
        # ========================================
        if progress_callback:
            progress_callback("reading_files")
            
        with profiler.time_block("A. Process Source PDF"):
            pdf_bytes_source = convert_pdf_bytes_to_bytes_by_pypdfium2(
                source_pdf_bytes,
                start_page_id=0,
                end_page_id=1
            )

        env_enabled = (
            flags["enable_270_only"] or 
            flags["enable_all_rotations"] or 
            flags["use_balance_id_strategy"] or 
            flags["target_all_pages"]
        )

        # ========================================
        # STAGE 2: Analyze source document (AFTER source preprocessing completes)
        # ========================================
        if progress_callback:
            progress_callback("analyzing_source")

        if env_enabled:
            print("\n🚀 Using environment-configured flow")

            # Extract and rotate target pages (lightweight work)
            pdf_doc = pdfium.PdfDocument(BytesIO(target_pdf_bytes))
            actual_page_count = len(pdf_doc)
            pdf_doc.close()

            print(f"\n📄 PDF Analysis:")
            print(f"   • Total pages in PDF: {actual_page_count}")
            print(f"   • Mode: Process ALL {actual_page_count} pages")

            with profiler.time_block("B. Extract Target Pages as Images"):
                page_imgs = get_page_images(target_pdf_bytes, dpi=300, max_pages=None)

            print(f"   • Extracted: {len(page_imgs)} image(s)")

            debug_dir = Path("ocr_debug_images")
            debug_dir.mkdir(exist_ok=True)

            print(f"\n💾 Saving original pages to {debug_dir}/")
            for idx, img in enumerate(page_imgs, start=1):
                img_path = debug_dir / f"step1_original_page_{idx}.png"
                img.save(img_path)
                print(f"   • Page {idx}: {img_path.name} ({img.width}x{img.height})")

            angles = determine_rotation_angles(flags)
            print(f"\n📐 Rotation Strategy:")
            print(f"   • Testing angles: {angles}")
            print(f"   • Strategy: {'BALANCE ID counting' if flags['use_balance_id_strategy'] else 'Default (first angle)'}")

            rotation_pdfs = []

            with profiler.time_block("C. Rotate & Merge Pages"):
                for angle in angles:
                    print(f"\n   🔄 Processing {angle}° rotation:")

                    rotated_pages = []
                    for page_idx, img in enumerate(page_imgs, start=1):
                        rotated_img = img.rotate(angle, expand=True)
                        rotated_pages.append(rotated_img)

                        rot_path = debug_dir / f"step2_rotated_{angle}deg_page_{page_idx}.png"
                        rotated_img.save(rot_path)
                        print(f"      • Page {page_idx}: rotated to {angle}° ({rotated_img.width}x{rotated_img.height})")

                    if len(rotated_pages) == 1:
                        merged_img = rotated_pages[0]
                        print(f"      • Single page - using as-is")
                    else:
                        merged_img = merge_rotated_pages(rotated_pages, angle)
                        print(f"      • Merged {len(rotated_pages)} pages into {merged_img.width}x{merged_img.height}")

                    merged_path = debug_dir / f"step3_final_{angle}deg_merged.png"
                    merged_img.save(merged_path)
                    print(f"      ✅ Final: {merged_path.name}")

                    rotated_pdf_bytes = image_to_pdf_bytes(merged_img)
                    rotation_pdfs.append((angle, rotated_pdf_bytes))

                    pdf_path = debug_dir / f"step4_ocr_input_{angle}deg.pdf"
                    with open(pdf_path, 'wb') as f:
                        f.write(rotated_pdf_bytes)
                    print(f"      💾 PDF: {pdf_path.name} ({len(rotated_pdf_bytes)} bytes)")

            # ========================================
            # STAGE 3: Analyze target document (AFTER rotation, BEFORE heavy OCR)
            # ========================================
            if progress_callback:
                progress_callback("analyzing_target")

            print(f"\n🔍 OCR Processing:")
            print(f"   • Running OCR on {len(rotation_pdfs)} rotation(s)...")

            all_pdf_bytes = [pdf_bytes for _, pdf_bytes in rotation_pdfs]
            all_langs = ["ch"] * len(all_pdf_bytes)

            # HEAVY OCR WORK HAPPENS HERE (60+ seconds)
            with profiler.time_block("D. Batch OCR Processing"):
                ocr_results = batch_ocr_process(all_pdf_bytes, all_langs, profiler)

            print(f"\n📝 Saving OCR results:")
            for idx, (angle, _) in enumerate(rotation_pdfs):
                text_path = debug_dir / f"step5_ocr_output_{angle}deg.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== OCR OUTPUT FOR {angle}° ROTATION ===\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Character count: {len(ocr_results[idx])}\n")
                    f.write("="*60 + "\n\n")
                    f.write(ocr_results[idx])

                preview = ocr_results[idx][:300].replace('\n', ' ')
                print(f"   • {angle}° ({len(ocr_results[idx])} chars): {preview}...")

            if flags["use_balance_id_strategy"] and len(angles) > 1:
                print(f"\n⚖️  Applying BALANCE ID Strategy:")

                scores = []
                for idx, (angle, pdf_bytes) in enumerate(rotation_pdfs):
                    ocr_text = ocr_results[idx]
                    balance_id_count = count_balance_ids_in_text(ocr_text)
                    scores.append((balance_id_count, angle, pdf_bytes, ocr_text))
                    print(f"   • {angle:3d}°: {balance_id_count} Balance ID(s) detected")

                scores.sort(reverse=True, key=lambda x: x[0])
                best_count, best_angle, best_pdf_bytes, best_ocr_text = scores[0]

                print(f"\n✅ Selected: {best_angle}° rotation ({best_count} Balance IDs)")

                ocr_text_target = best_ocr_text
            else:
                best_angle, best_pdf_bytes = rotation_pdfs[0]
                ocr_text_target = ocr_results[0]
                print(f"\n✅ Using default: {best_angle}° rotation")

            print("\n📝 Processing source document...")
            with profiler.time_block("E. OCR Source Document"):
                ocr_text_source = batch_ocr_process(
                    [pdf_bytes_source],
                    ["ch"],
                    profiler
                )[0]

        else:
            # Default flow (not used with your current config)
            print("\n📌 Using default flow (first 2 pages, 270° only)")

            with profiler.time_block("B. Extract Target Pages as Images"):
                page_imgs = get_page_images(target_pdf_bytes, dpi=300, max_pages=2)

            rotation_pdfs = []
            with profiler.time_block("C. Rotate & Merge Pages"):
                for angle in [270]:
                    merged_img = merge_rotated_pages(page_imgs, angle)
                    rotated_pdf_bytes = image_to_pdf_bytes(merged_img)
                    rotation_pdfs.append((angle, rotated_pdf_bytes))
                    merged_img.save(f"rotated_{angle}deg_merged.png")
                    print(f"   💾 Saved: rotated_{angle}deg_merged.png ({merged_img.width}x{merged_img.height})")

            # STAGE 3 for default flow
            if progress_callback:
                progress_callback("analyzing_target")

            print("\n" + "-"*60)
            print("BATCH OCR INFERENCE")
            print("-"*60 + "\n")

            all_pdf_bytes = [pdf_bytes_source]
            all_langs = ["ch"]

            for angle, pdf_bytes in rotation_pdfs:
                all_pdf_bytes.append(pdf_bytes)
                all_langs.append("ch")

            with profiler.time_block("D. Batch OCR Processing"):
                ocr_results = batch_ocr_process(all_pdf_bytes, all_langs, profiler)

            ocr_text_source = ocr_results[0]
            ocr_text_target = ocr_results[1] if len(ocr_results) > 1 else ""

        # ========================================
        # STAGE 4: Validate results (AFTER all OCR completes)
        # ========================================
        print("\n🔎 Validating documents...")
        if progress_callback:
            progress_callback("validating_results")
            
        with profiler.time_block("F. Validation"):
            validation_result = validate_document_pair(ocr_text_source, ocr_text_target)

        profiler.print_summary()

        return validation_result

    # At the end of process_ocr_validation function
    except Exception as e:
        print(f"\n❌ Error during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return full structure even on error
        return {
            "source_ocr_result": {
                "instrument_id": None,
                "target_weight_g": None,
                "weight_range": {"min_g": None, "max_g": None},
                "used_from": None,
                "used_upto": None
            },
            "target_ocr_result": {
                "weighing_protocols": []
            },
            "validation": {
                "weight_validated": False,
                "datetime_validated": False,
                "overall_validated": False
            },
            "error": f"OCR processing failed: {str(e)}"
        }