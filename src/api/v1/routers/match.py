"""
Redesigned Match router for the Resume Sorter API
Uses a session-based approach to separate job description and resume uploads
"""
import uuid
import logging
import os
import shutil
from datetime import datetime, timedelta
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import tempfile

from src.api.v1.schemas import JobDesc, MatchResponse, BatchProcessResponse, ErrorResponse
from src.services.parser import ResumeParser
from src.services.matcher import ResumeMatcher
from src.config import get_settings

# Configure router
router = APIRouter(
    prefix="/match",
    tags=["Resume Matching"],
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)

# Configure logging
logger = logging.getLogger(__name__)

# In-memory storage for job sessions (would use a database in production)
# Structure: {session_id: {"job": JobDesc, "files": [filename1, filename2, ...], "status": "created|processing|complete", "results": []}}
job_sessions: Dict[str, Dict] = {}

# Temporary directory for file storage
TEMP_DIR = tempfile.gettempdir()

# Helper functions
def create_session_dir(session_id: str) -> str:
    """Create a directory for session files"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in job_sessions.items():
        if current_time - session_data.get("created_at", current_time) > timedelta(hours=1):
            sessions_to_remove.append(session_id)
            # Remove associated directory
            session_dir = os.path.join(TEMP_DIR, session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
    
    for session_id in sessions_to_remove:
        del job_sessions[session_id]

# API Endpoints
@router.post(
    "/sessions", 
    response_model=dict, 
    summary="Create a new job matching session"
)
async def create_session():
    """
    Create a new session for job matching.
    
    Returns a session ID that will be used in subsequent calls.
    """
    session_id = f"session_{uuid.uuid4().hex[:10]}"
    job_sessions[session_id] = {
        "created_at": datetime.now(),
        "status": "created",
        "files": []
    }
    create_session_dir(session_id)
    
    return {"session_id": session_id}


@router.post(
    "/sessions/{session_id}/job", 
    response_model=dict,
    summary="Add job description to a session"
)
async def add_job_description(
    session_id: str = Path(..., description="Session ID"),
    job: JobDesc = Depends()
):
    """
    Add a job description to an existing session.
    
    - **session_id**: ID of the previously created session
    - **job**: Job description details
    
    Returns confirmation of the job description being added.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    job_sessions[session_id]["job"] = job
    
    return {"status": "success", "message": "Job description added successfully"}


@router.post(
    "/sessions/{session_id}/resumes", 
    response_model=dict,
    summary="Upload resumes to a session"
)
async def upload_resumes(
    session_id: str = Path(..., description="Session ID"),
    files: List[UploadFile] = File(...),
    settings=Depends(get_settings)
):
    """
    Upload resumes to an existing session.
    
    - **session_id**: ID of the previously created session
    - **files**: List of resume files to analyze
    
    Returns confirmation of the number of resumes uploaded.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_dir = os.path.join(TEMP_DIR, session_id)
    
    # Validate file extensions
    uploaded_files = []
    for file in files:
        ext = file.filename.split('.')[-1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {ext}. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Save file to session directory
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            if len(content) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} exceeds maximum size of {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
                )
            f.write(content)
        
        uploaded_files.append(file.filename)
    
    # Add files to session
    job_sessions[session_id]["files"].extend(uploaded_files)
    
    return {
        "status": "success", 
        "message": f"{len(uploaded_files)} resumes uploaded", 
        "total_files": len(job_sessions[session_id]["files"])
    }


@router.post(
    "/sessions/{session_id}/process", 
    response_model=BatchProcessResponse,
    summary="Start processing a session"
)
async def process_session(
    background_tasks: BackgroundTasks,
    session_id: str = Path(..., description="Session ID")
):
    """
    Start processing resumes in a session.
    
    - **session_id**: ID of the session to process
    
    Returns a job ID and status information.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = job_sessions[session_id]
    
    # Check if job description exists
    if "job" not in session:
        raise HTTPException(status_code=400, detail="No job description found in session")
    
    # Check if files exist
    if not session["files"]:
        raise HTTPException(status_code=400, detail="No resumes found in session")
    
    # Update session status
    session["status"] = "processing"
    
    # Add background task
    background_tasks.add_task(
        process_session_files,
        session_id=session_id
    )
    
    return BatchProcessResponse(
        job_id=session_id,
        status="processing",
        total_resumes=len(session["files"]),
        estimated_completion=datetime.now() + timedelta(minutes=1)  # Estimate completion time
    )


@router.get(
    "/sessions/{session_id}/results", 
    response_model=List[MatchResponse],
    summary="Get processing results"
)
async def get_results(session_id: str = Path(..., description="Session ID")):
    """
    Get the results from a processed session.
    
    - **session_id**: ID of the processed session
    
    Returns the list of matches from the session.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = job_sessions[session_id]
    
    if session["status"] == "processing":
        raise HTTPException(status_code=102, detail="Processing in progress")
    
    if session["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Session is in state: {session['status']}")
    
    return session.get("results", [])


@router.get(
    "/sessions/{session_id}/status", 
    response_model=dict,
    summary="Check session status"
)
async def get_session_status(session_id: str = Path(..., description="Session ID")):
    """
    Check the status of a session.
    
    - **session_id**: ID of the session
    
    Returns the current status of the session.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = job_sessions[session_id]
    
    return {
        "status": session["status"],
        "files_count": len(session["files"]),
        "job_description": "present" if "job" in session else "missing",
        "created_at": session["created_at"]
    }


@router.delete(
    "/sessions/{session_id}", 
    response_model=dict,
    summary="Delete a session"
)
async def delete_session(session_id: str = Path(..., description="Session ID")):
    """
    Delete a session and all associated files.
    
    - **session_id**: ID of the session to delete
    
    Returns confirmation of deletion.
    """
    if session_id not in job_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Delete session directory
    session_dir = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    
    # Remove session from storage
    del job_sessions[session_id]
    
    return {"status": "success", "message": f"Session {session_id} deleted"}


# Background task function (not directly exposed as endpoint)
async def process_session_files(session_id: str):
    """Background task for processing resumes in a session"""
    try:
        session = job_sessions[session_id]
        job = session["job"]
        session_dir = os.path.join(TEMP_DIR, session_id)
        
        # Initialize services
        parser = ResumeParser()
        matcher = ResumeMatcher()
        
        # Parse resumes
        parsed_resumes = []
        for filename in session["files"]:
            file_path = os.path.join(session_dir, filename)
            with open(file_path, "rb") as f:
                content = f.read()
            
            parsed = await parser.parse_resume(content, filename)
            parsed_resumes.append(parsed)
        
        # Match resumes against job description
        results = await matcher.rank_resumes(job, parsed_resumes)
        
        # Update session with results
        session["results"] = results
        session["status"] = "complete"
        
    except Exception as e:
        logger.exception(f"Error in session processing {session_id}: {str(e)}")
        session = job_sessions.get(session_id)
        if session:
            session["status"] = "error"
            session["error_message"] = str(e)


# Run cleanup of old sessions periodically (would use a proper task scheduler in production)
@router.on_event("startup")
async def on_startup():
    """Run cleanup task on startup"""
    cleanup_old_sessions()