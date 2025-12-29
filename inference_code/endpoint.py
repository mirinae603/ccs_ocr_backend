from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional, List
import uuid
import asyncio

# -------------------------
# Mock functions — replace with yours
# -------------------------
async def extract_markdown_from_pdf(pdf_bytes: bytes) -> str:
    await asyncio.sleep(1)  # simulate OCR time
    return pdf_bytes.decode(errors="ignore")[:500]  # dummy text


def validate_documents(doc1: str, doc2: str):
    # your real validation logic goes here
    return {
        "instrument_id": "QC-BAL-006",
        "target_weight_g": "12.50",
        "records_checked": 3,
        "matching_records": [],
        "validated": True,
        "anomaly": False,
        "error": None
    }


# -------------------------
# Schemas
# -------------------------

class MatchingRecord(BaseModel):
    balance_id: str
    maximum_g: str
    matches_target: bool


class ValidationResult(BaseModel):
    instrument_id: Optional[str]
    target_weight_g: Optional[str]
    records_checked: int
    matching_records: List[MatchingRecord] = []
    validated: bool
    anomaly: bool
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    error: Optional[str] = None


# -------------------------
# In-memory job store
# -------------------------
jobs: Dict[str, Dict] = {}


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI()


# -------------------------
# Background worker
# -------------------------
async def process_job(job_id: str, doc1_bytes: bytes, doc2_bytes: bytes):
    try:
        jobs[job_id]["status"] = "processing"

        md1 = await extract_markdown_from_pdf(doc1_bytes)
        md2 = await extract_markdown_from_pdf(doc2_bytes)

        result = validate_documents(md1, md2)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)



# -------------------------
# 1️⃣ Create Job (upload + trigger)
# -------------------------
@app.post("/jobs", response_model=JobStatus)
async def create_job(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None
    }

    # READ FILES NOW
    source_bytes = await source.read()
    target_bytes = await target.read()

    # PASS BYTES — NOT UploadFile
    background_tasks.add_task(
        process_job,
        job_id,
        source_bytes,
        target_bytes
    )

    return JobStatus(job_id=job_id, status="queued")



# -------------------------
# 2️⃣ Check Job Status
# -------------------------
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):

    if job_id not in jobs:
        return JobStatus(job_id=job_id, status="not_found")

    job = jobs[job_id]
    return JobStatus(job_id=job_id, status=job["status"], error=job["error"])


# -------------------------
# 3️⃣ Get Job Result
# -------------------------
@app.get("/jobs/{job_id}/result", response_model=ValidationResult)
async def get_job_result(job_id: str):

    if job_id not in jobs:
        raise ValueError("Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        return ValidationResult(
            instrument_id=None,
            target_weight_g=None,
            records_checked=0,
            matching_records=[],      # 👈 IMPORTANT — include this
            validated=False,
            anomaly=False,
            error="Job not finished yet"
        )

    return job["result"]

