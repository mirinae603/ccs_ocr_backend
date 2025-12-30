import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional, List
import uuid
import tempfile
from pathlib import Path

from ocr_inference_utils_code import parse_doc
from regex_utils_code import validate_documents


# =========================
# Schemas
# =========================

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


# =========================
# In-memory job store
# =========================
jobs: Dict[str, Dict] = {}


# =========================
# FastAPI App
# =========================
app = FastAPI()


# =========================
# Background Job
# =========================
async def process_job(job_id: str, doc1_bytes: bytes, doc2_bytes: bytes):

    try:
        jobs[job_id]["status"] = "processing"

        # 1️⃣ temp dir
        tmp_dir = Path(tempfile.mkdtemp())

        doc1_path = tmp_dir / "source.pdf"
        doc2_path = tmp_dir / "target.pdf"
        output_dir = tmp_dir / "results"

        # 2️⃣ save uploaded PDFs
        doc1_path.write_bytes(doc1_bytes)
        doc2_path.write_bytes(doc2_bytes)

        # 3️⃣ run MinerU OCR
        parse_doc(
            path_list=[doc1_path, doc2_path],
            output_dir=str(output_dir),
            backend="pipeline",
            method="auto",
        )

        # 4️⃣ find Markdown files that were generated
        md_files = list(output_dir.rglob("*.md"))
        if len(md_files) != 2:
            raise RuntimeError("Expected 2 markdown files but found: " + str(len(md_files)))

        # consistent ordering
        md_files.sort()

        md1 = md_files[0].read_text(encoding="utf-8")
        md2 = md_files[1].read_text(encoding="utf-8")

        # 5️⃣ run REAL validation
        result = validate_documents(md1, md2)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# =========================
# 1️⃣ Create Job
# =========================
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
        "error": None,
    }

    # read bytes NOW so UploadFile doesn't close
    source_bytes = await source.read()
    target_bytes = await target.read()

    background_tasks.add_task(
        process_job,
        job_id,
        source_bytes,
        target_bytes,
    )

    return JobStatus(job_id=job_id, status="queued")


# =========================
# 2️⃣ Check Job Status
# =========================
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):

    if job_id not in jobs:
        return JobStatus(job_id=job_id, status="not_found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        error=job["error"]
    )


# =========================
# 3️⃣ Get Job Result
# =========================
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
            matching_records=[],
            validated=False,
            anomaly=False,
            error="Job not finished yet",
        )

    return job["result"]
