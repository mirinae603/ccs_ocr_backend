import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import uuid
import asyncio
import shutil
import time
from ocr_inference import process_ocr_validation
from file_database import get_job_manager, close_database
from contextlib import asynccontextmanager

ENABLE_270_ONLY_ROTATION = False
ENABLE_ALL_ROTATIONS = True
USE_BALANCE_ID_STRATEGY = True
TARGET_ALL_PAGES = True

# =========================
# Storage Configuration
# =========================
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

# =========================
# Lifespan Context Manager
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)"""
    # Startup
    print("🚀 Starting up FastAPI application...")
    print(f"📁 PDF storage directory: {UPLOAD_DIR.absolute()}")
    yield
    # Shutdown
    print("🛑 Shutting down FastAPI application...")
    close_database()

# =========================
# FastAPI App
# =========================

app = FastAPI(lifespan=lifespan)

# =========================
# Mount Static Files for PDF Access
# =========================
app.mount("/pdfs", StaticFiles(directory=str(UPLOAD_DIR)), name="pdfs")

# =========================
# CORS CONFIG (ALLOW ALL)
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Schemas - UPDATED to include all fields
# =========================

class WeighingProtocol(BaseModel):
    balance_id: str
    matches_instrument_id: bool
    weighing_datetime: Optional[str] = None
    maximum_g: str
    weight_in_range: bool
    weight_matches_target: bool
    weight_difference: str
    datetime_in_used_range: Optional[bool] = None

class SourceOcrResult(BaseModel):
    instrument_id: str  # Can be empty string if not found
    target_weight_g: str  # Can be empty string if not found
    weight_range: Dict[str, str]
    used_from: str  # Can be empty string if not found
    used_upto: str  # Can be empty string if not found



class TargetOcrResult(BaseModel):
    weighing_protocols: List[WeighingProtocol]

class ValidationDetails(BaseModel):
    weight_validated: bool
    datetime_validated: bool
    overall_validated: bool

class ValidationResult(BaseModel):
    source_ocr_result: SourceOcrResult
    target_ocr_result: TargetOcrResult
    validation: ValidationDetails
    source_pdf_url: Optional[str] = None
    target_pdf_url: Optional[str] = None
    error: Optional[str] = None  # ← ADD THIS to include error message


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    error: Optional[str] = None
    result: Optional[ValidationResult] = None
    source_pdf_url: Optional[str] = None
    target_pdf_url: Optional[str] = None

class LastJobResponse(BaseModel):
    last_job_id: Optional[str] = None

class JobStats(BaseModel):
    total_jobs: int
    stats_by_status: Dict[str, int]

# =========================
# Helper Functions
# =========================

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to disk asynchronously"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        await upload_file.close()

def get_pdf_url(filename: str, base_url: str = "http://35.200.192.66:8000") -> str:
    """Generate URL for accessing stored PDF"""
    return f"{base_url}/pdfs/{filename}"

# =========================
# Background Job Processing with Time-Spaced Status Updates
# =========================

async def process_job(
    job_id: str,
    source_path: Path,
    target_path: Path,
    source_pdf_url: str,
    target_pdf_url: str
):
    """Process OCR validation job with real-time status updates via callback"""
    job_mgr = get_job_manager()
    
    def progress_callback(status: str):
        """Called from OCR process to update job status"""
        job_mgr.update_job_status(job_id, status)
        print(f"📊 Job {job_id} → {status}")
    
    try:
        # Read PDF files
        with open(source_path, "rb") as f:
            source_bytes = f.read()
        
        with open(target_path, "rb") as f:
            target_bytes = f.read()
        
        # Run OCR with progress callback
        loop = asyncio.get_event_loop()
        validation_result = await loop.run_in_executor(
            None, 
            process_ocr_validation, 
            source_bytes, 
            target_bytes,
            progress_callback  
        )
        
        # Add PDF URLs to validation result
        validation_result["source_pdf_url"] = source_pdf_url
        validation_result["target_pdf_url"] = target_pdf_url
        
        # Update job with result (status becomes "completed")
        job_mgr.update_job_result(job_id, validation_result)
        
        print(f"✅ Job {job_id} completed successfully")

    except Exception as e:
        job_mgr.update_job_error(job_id, str(e))
        print(f"❌ Job {job_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()


def convert_db_job_to_response(job: Dict) -> JobResponse:
    """Convert database job document to JobResponse with FULL data and error handling"""
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Convert result if present
    result = None
    source_pdf_url = job.get("source_pdf_url")
    target_pdf_url = job.get("target_pdf_url")
    
    if job.get("result") and job["status"] == "completed":
        result_data = job["result"]
        
        # Extract PDF URLs from result if not in job metadata
        if not source_pdf_url:
            source_pdf_url = result_data.get("source_pdf_url")
        if not target_pdf_url:
            target_pdf_url = result_data.get("target_pdf_url")
        
        # Check if we have the expected structure
        if "source_ocr_result" in result_data and "target_ocr_result" in result_data:
            try:
                # Safely build SourceOcrResult with None handling
                source_data = result_data["source_ocr_result"]
                source_ocr = SourceOcrResult(
                    instrument_id=source_data.get("instrument_id") or "",
                    target_weight_g=source_data.get("target_weight_g") or "",
                    weight_range=source_data.get("weight_range", {"min_g": "1", "max_g": "2"}),
                    used_from=source_data.get("used_from") or "",
                    used_upto=source_data.get("used_upto") or ""
                )
                
                # Safely build TargetOcrResult
                target_data = result_data["target_ocr_result"]
                weighing_protocols = []
                for protocol in target_data.get("weighing_protocols", []):
                    try:
                        weighing_protocols.append(WeighingProtocol(**protocol))
                    except Exception as e:
                        print(f"⚠️  Skipping invalid protocol: {e}")
                        continue
                
                target_ocr = TargetOcrResult(weighing_protocols=weighing_protocols)
                
                # Safely build ValidationDetails
                validation_data = result_data.get("validation", {})
                validation = ValidationDetails(
                    weight_validated=validation_data.get("weight_validated", False),
                    datetime_validated=validation_data.get("datetime_validated", False),
                    overall_validated=validation_data.get("overall_validated", False)
                )
                
                # Build full result
                result = ValidationResult(
                    source_ocr_result=source_ocr,
                    target_ocr_result=target_ocr,
                    validation=validation,
                    source_pdf_url=source_pdf_url,
                    target_pdf_url=target_pdf_url
                )
                
            except Exception as e:
                print(f"❌ Error building ValidationResult: {e}")
                import traceback
                traceback.print_exc()
                # Result will remain None, error will be included below

    # Build response with error handling
    error = job.get("error")
    if job.get("result") and "error" in job["result"]:
        # Combine job error and result error
        result_error = job["result"]["error"]
        error = f"{error}; {result_error}" if error else result_error

    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=None,
        error=error,
        result=result,
        source_pdf_url=source_pdf_url,
        target_pdf_url=target_pdf_url
    )


# =========================
# 1️⃣ Create Job
# =========================

@app.post("/jobs", response_model=JobResponse)
async def create_job(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    job_mgr = get_job_manager()
    job_id = str(uuid.uuid4())

    # Generate unique filenames
    source_filename = f"{job_id}_source_{source.filename}"
    target_filename = f"{job_id}_target_{target.filename}"
    
    source_path = UPLOAD_DIR / source_filename
    target_path = UPLOAD_DIR / target_filename

    # Save files to disk
    await save_upload_file(source, source_path)
    await save_upload_file(target, target_path)

    # Generate PDF URLs
    source_pdf_url = get_pdf_url(source_filename)
    target_pdf_url = get_pdf_url(target_filename)

    # Create job in database with PDF URLs
    job_mgr.create_job(
        job_id=job_id,
        source_filename=source.filename,
        target_filename=target.filename,
        status="processing_initiated",
        source_pdf_url=source_pdf_url,
        target_pdf_url=target_pdf_url
    )

    # Start background processing
    asyncio.create_task(
        process_job(job_id, source_path, target_path, source_pdf_url, target_pdf_url)
    )

    return JobResponse(
        job_id=job_id,
        status="processing_initiated",
        progress=None,
        error=None,
        result=None,
        source_pdf_url=source_pdf_url,
        target_pdf_url=target_pdf_url
    )

# =========================
# 2️⃣ Get Last Job (Refresh Recovery)
# =========================

@app.get("/jobs/last", response_model=LastJobResponse)
async def get_last_job():
    job_mgr = get_job_manager()
    last_job_id = job_mgr.get_last_job_id()
    return LastJobResponse(last_job_id=last_job_id)

# =========================
# 3️⃣ Get Job Status / Result
# =========================

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job_mgr = get_job_manager()
    job = job_mgr.get_job(job_id)

    if not job:
        return JobResponse(
            job_id=job_id,
            status="not_found",
            error="Job not found",
        )

    return convert_db_job_to_response(job)

# =========================
# 4️⃣ Additional Endpoints (Bonus)
# =========================

@app.get("/jobs", response_model=List[JobResponse])
async def get_all_jobs(limit: int = 100, skip: int = 0):
    """Get all jobs with pagination"""
    job_mgr = get_job_manager()
    jobs = job_mgr.get_all_jobs(limit=limit, skip=skip)
    return [convert_db_job_to_response(job) for job in jobs]

@app.get("/jobs/status/{status}", response_model=List[JobResponse])
async def get_jobs_by_status(status: str, limit: int = 100):
    """Get jobs filtered by status"""
    job_mgr = get_job_manager()
    jobs = job_mgr.get_jobs_by_status(status, limit=limit)
    return [convert_db_job_to_response(job) for job in jobs]

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and associated PDF files"""
    job_mgr = get_job_manager()
    job = job_mgr.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete associated PDF files
    source_filename = f"{job_id}_source_{job.get('source_filename', '')}"
    target_filename = f"{job_id}_target_{job.get('target_filename', '')}"
    
    source_path = UPLOAD_DIR / source_filename
    target_path = UPLOAD_DIR / target_filename
    
    if source_path.exists():
        source_path.unlink()
    if target_path.exists():
        target_path.unlink()
    
    # Delete job from database
    success = job_mgr.delete_job(job_id)

    if success:
        return {"message": f"Job {job_id} and associated files deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/stats", response_model=JobStats)
async def get_stats():
    """Get job statistics"""
    job_mgr = get_job_manager()
    total = job_mgr.get_job_count()
    stats = job_mgr.get_stats()

    return JobStats(
        total_jobs=total,
        stats_by_status=stats
    )