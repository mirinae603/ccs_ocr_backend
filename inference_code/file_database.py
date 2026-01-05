import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from filelock import FileLock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# File-Based Storage Configuration
# =========================

class FileDatabase:
    def __init__(self, data_dir: str = "data"):
        """Initialize file-based database"""
        self.data_dir = Path(data_dir)
        self.jobs_file = self.data_dir / "jobs.json"
        self.lock_file = self.data_dir / "jobs.lock"
        self.lock = threading.Lock()

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize jobs file if it doesn't exist
        if not self.jobs_file.exists():
            self._write_jobs({})
            logger.info(f"✅ Created new jobs database at {self.jobs_file}")
        else:
            logger.info(f"✅ Loaded existing jobs database from {self.jobs_file}")

        self.connected = True

    def _read_jobs(self) -> Dict:
        """Read jobs from file with file locking"""
        try:
            with FileLock(str(self.lock_file), timeout=5):
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert string dates back to datetime objects for consistency
                    for job_id, job in data.items():
                        if 'created_at' in job and isinstance(job['created_at'], str):
                            job['created_at'] = datetime.fromisoformat(job['created_at'])
                        if 'updated_at' in job and isinstance(job['updated_at'], str):
                            job['updated_at'] = datetime.fromisoformat(job['updated_at'])
                    return data
        except json.JSONDecodeError:
            logger.error("❌ Corrupted jobs file, creating new one")
            return {}
        except Exception as e:
            logger.error(f"❌ Error reading jobs file: {e}")
            return {}

    def _write_jobs(self, jobs: Dict) -> bool:
        """Write jobs to file with file locking and atomic writes"""
        try:
            # Convert datetime objects to ISO format strings for JSON serialization
            serializable_jobs = {}
            for job_id, job in jobs.items():
                job_copy = job.copy()
                if 'created_at' in job_copy and isinstance(job_copy['created_at'], datetime):
                    job_copy['created_at'] = job_copy['created_at'].isoformat()
                if 'updated_at' in job_copy and isinstance(job_copy['updated_at'], datetime):
                    job_copy['updated_at'] = job_copy['updated_at'].isoformat()
                serializable_jobs[job_id] = job_copy

            with FileLock(str(self.lock_file), timeout=5):
                # Write to temporary file first (atomic write)
                temp_file = self.jobs_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_jobs, f, indent=2, ensure_ascii=False)

                # Replace original file with temp file
                temp_file.replace(self.jobs_file)

            return True
        except Exception as e:
            logger.error(f"❌ Error writing jobs file: {e}")
            return False

    def close(self):
        """Close database (no-op for file-based storage)"""
        logger.info("File database closed")


# =========================
# Job Operations
# =========================

class JobManager:
    def __init__(self, db: FileDatabase):
        self.db = db
        self.last_job_cache = None
        self._load_last_job_cache()

    def _load_last_job_cache(self):
        """Load last job ID from database on startup"""
        jobs = self.db._read_jobs()
        if jobs:
            sorted_jobs = sorted(
                jobs.values(),
                key=lambda x: x.get('created_at', datetime.min),
                reverse=True
            )
            if sorted_jobs:
                self.last_job_cache = sorted_jobs[0]['job_id']

    def create_job(
        self, 
        job_id: str, 
        source_filename: str, 
        target_filename: str,
        status: str = "processing_initiated",
        source_pdf_url: Optional[str] = None,
        target_pdf_url: Optional[str] = None
    ) -> Dict:
        """Create a new job record with PDF URLs"""
        with self.db.lock:
            jobs = self.db._read_jobs()

            if job_id in jobs:
                logger.error(f"❌ Job {job_id} already exists")
                raise ValueError(f"Job {job_id} already exists")

            job_doc = {
                "job_id": job_id,
                "status": status,
                "source_filename": source_filename,
                "target_filename": target_filename,
                "source_pdf_url": source_pdf_url,
                "target_pdf_url": target_pdf_url,
                "result": None,
                "error": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            jobs[job_id] = job_doc

            if self.db._write_jobs(jobs):
                self.last_job_cache = job_id
                logger.info(f"📝 Job created: {job_id}")
                logger.info(f"   📄 Source PDF: {source_pdf_url}")
                logger.info(f"   📄 Target PDF: {target_pdf_url}")
                return job_doc
            else:
                raise Exception("Failed to write job to database")

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        jobs = self.db._read_jobs()
        job = jobs.get(job_id)

        if job:
            logger.info(f"📖 Job retrieved: {job_id}")
        else:
            logger.warning(f"⚠️  Job not found: {job_id}")

        return job

    def update_job_status(self, job_id: str, status: str) -> bool:
        """Update job status"""
        with self.db.lock:
            jobs = self.db._read_jobs()

            if job_id not in jobs:
                logger.warning(f"⚠️  Job {job_id} not found for status update")
                return False

            jobs[job_id]["status"] = status
            jobs[job_id]["updated_at"] = datetime.utcnow()

            if self.db._write_jobs(jobs):
                logger.info(f"🔄 Job {job_id} status updated: {status}")
                return True
            else:
                logger.error(f"❌ Failed to update job {job_id} status")
                return False

    def update_job_result(self, job_id: str, result: Dict) -> bool:
        """
        Update job with validation result - PRESERVING ALL DATA
        """
        with self.db.lock:
            jobs = self.db._read_jobs()

            if job_id not in jobs:
                logger.warning(f"⚠️  Job {job_id} not found for result update")
                return False

            # Preserve PDF URLs from job creation if not in result
            if "source_pdf_url" not in result and jobs[job_id].get("source_pdf_url"):
                result["source_pdf_url"] = jobs[job_id]["source_pdf_url"]
            
            if "target_pdf_url" not in result and jobs[job_id].get("target_pdf_url"):
                result["target_pdf_url"] = jobs[job_id]["target_pdf_url"]

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result  # Store the FULL result
            jobs[job_id]["updated_at"] = datetime.utcnow()

            if self.db._write_jobs(jobs):
                logger.info(f"✅ Job {job_id} completed with result")
                logger.info(f"   ✓ Validated: {result.get('validation', {}).get('overall_validated', False)}")
                return True
            else:
                logger.error(f"❌ Failed to update job {job_id} result")
                return False

    def update_job_error(self, job_id: str, error: str) -> bool:
        """Update job with error"""
        with self.db.lock:
            jobs = self.db._read_jobs()

            if job_id not in jobs:
                logger.warning(f"⚠️  Job {job_id} not found for error update")
                return False

            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error
            jobs[job_id]["updated_at"] = datetime.utcnow()

            if self.db._write_jobs(jobs):
                logger.error(f"❌ Job {job_id} failed: {error}")
                return True
            else:
                logger.error(f"❌ Failed to update job {job_id} error")
                return False

    def get_last_job_id(self) -> Optional[str]:
        """Get the most recent job ID"""
        # Check cache first
        if self.last_job_cache:
            return self.last_job_cache

        jobs = self.db._read_jobs()
        if not jobs:
            logger.info("No jobs found in database")
            return None

        # Find most recent job
        sorted_jobs = sorted(
            jobs.values(),
            key=lambda x: x.get('created_at', datetime.min),
            reverse=True
        )

        if sorted_jobs:
            self.last_job_cache = sorted_jobs[0]['job_id']
            logger.info(f"📌 Last job ID: {self.last_job_cache}")
            return self.last_job_cache

        return None

    def get_all_jobs(self, limit: int = 100, skip: int = 0) -> List[Dict]:
        """Get all jobs with pagination"""
        jobs = self.db._read_jobs()

        # Sort by created_at descending
        sorted_jobs = sorted(
            jobs.values(),
            key=lambda x: x.get('created_at', datetime.min),
            reverse=True
        )

        # Apply pagination
        paginated_jobs = sorted_jobs[skip:skip + limit]

        logger.info(f"📚 Retrieved {len(paginated_jobs)} jobs (total: {len(sorted_jobs)})")
        return paginated_jobs

    def get_jobs_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Get jobs filtered by status"""
        jobs = self.db._read_jobs()

        # Filter by status
        filtered_jobs = [
            job for job in jobs.values()
            if job.get('status') == status
        ]

        # Sort by created_at descending
        sorted_jobs = sorted(
            filtered_jobs,
            key=lambda x: x.get('created_at', datetime.min),
            reverse=True
        )

        # Apply limit
        limited_jobs = sorted_jobs[:limit]

        logger.info(f"📊 Retrieved {len(limited_jobs)} jobs with status: {status}")
        return limited_jobs

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        with self.db.lock:
            jobs = self.db._read_jobs()

            if job_id not in jobs:
                logger.warning(f"⚠️  Job not found for deletion: {job_id}")
                return False

            # Log PDF URLs before deletion for reference
            job = jobs[job_id]
            if job.get("source_pdf_url") or job.get("target_pdf_url"):
                logger.info(f"🗑️  Deleting job with PDFs:")
                logger.info(f"   📄 Source: {job.get('source_pdf_url')}")
                logger.info(f"   📄 Target: {job.get('target_pdf_url')}")

            del jobs[job_id]

            if self.db._write_jobs(jobs):
                logger.info(f"🗑️  Job deleted: {job_id}")

                # Clear cache if it was the last job
                if self.last_job_cache == job_id:
                    self.last_job_cache = None

                return True
            else:
                logger.error(f"❌ Failed to delete job {job_id}")
                return False

    def get_job_count(self) -> int:
        """Get total number of jobs"""
        jobs = self.db._read_jobs()
        count = len(jobs)
        logger.info(f"📊 Total jobs: {count}")
        return count

    def get_stats(self) -> Dict:
        """Get job statistics"""
        jobs = self.db._read_jobs()

        stats = {}
        for job in jobs.values():
            status = job.get('status', 'unknown')
            stats[status] = stats.get(status, 0) + 1

        logger.info(f"📊 Job stats: {stats}")
        return stats


# =========================
# Global Database Instance
# =========================

# Initialize file-based database
try:
    file_db = FileDatabase(data_dir="data")
    job_manager = JobManager(file_db)
    logger.info("✅ File-based database manager initialized successfully")
    logger.info(f"📁 Data stored in: {file_db.jobs_file.absolute()}")
except Exception as e:
    logger.error(f"❌ Failed to initialize file database: {e}")
    raise


# =========================
# Utility Functions
# =========================

def get_job_manager() -> JobManager:
    """Get the global job manager instance"""
    return job_manager


def close_database():
    """Close database connection (call on shutdown)"""
    file_db.close()

