import re
from decimal import Decimal
from typing import List, Dict, Optional



def normalize_text(md: str) -> str:
 
    md = re.sub(r"<[^>]+>", " ", md)
    md = re.sub(r"\s+", " ", md)
    return md.strip()



def extract_instrument_id_doc1(text: str) -> Optional[str]:
    
    match = re.search(
        r"Instrument\s*\(Instrument ID\).*?\(\s*([A-Z0-9\-]+)\s*\)",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def extract_target_weight_doc1(text: str) -> Optional[Decimal]:
   
    match = re.search(
        r"about\s+([\d]+\.\d+)\s*\(.*?\)\s*g",
        text,
        re.IGNORECASE,
    )
    return Decimal(match.group(1)) if match else None



def split_weighing_blocks(doc2: str) -> List[str]:
    
    blocks = re.split(
        r"WEIGHING\s+PROTOCOL",
        doc2,
        flags=re.IGNORECASE,
    )
    return [b.strip() for b in blocks if b.strip()]


def extract_balance_id_from_block(block: str) -> Optional[str]:
   
    match = re.search(
        r"Balance\s*ID\s*[:\-]?\s*([A-Z0-9\-]+)",
        block,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def extract_maximum_from_block(block: str) -> Optional[Decimal]:
   
    match = re.search(
        r"Maximum\s+([\d]+\.\d+)",
        block,
        re.IGNORECASE,
    )
    return Decimal(match.group(1)) if match else None



def validate_weight(
    target: Decimal,
    measured: Decimal,
    tolerance: Decimal = Decimal("0.0001"),
) -> bool:
    
    return abs(target - measured) <= tolerance



def validate_documents(
    markdown_text_doc1: str,
    markdown_text_doc2: str,
) -> Dict:

    
    doc1 = normalize_text(markdown_text_doc1)
    doc2 = normalize_text(markdown_text_doc2)

   
    instrument_id = extract_instrument_id_doc1(doc1)
    target_weight = extract_target_weight_doc1(doc1)

    if not instrument_id or target_weight is None:
        return {
            "validated": False,
            "error": "Instrument ID or target weight missing in Doc-1",
        }

  
    blocks = split_weighing_blocks(doc2)

    records = []

    for block in blocks:
        balance_id = extract_balance_id_from_block(block)
        if balance_id != instrument_id:
            continue

        max_weight = extract_maximum_from_block(block)
        if max_weight is None:
            continue

        records.append({
            "balance_id": balance_id,
            "maximum_g": str(max_weight),
            "matches_target": validate_weight(target_weight, max_weight),
        })

    valid_match = any(r["matches_target"] for r in records)

    return {
        "instrument_id": instrument_id,
        "target_weight_g": str(target_weight),
        "records_checked": len(records),
        "matching_records": records,
        "validated": valid_match,
        "anomaly": not valid_match,
    }

