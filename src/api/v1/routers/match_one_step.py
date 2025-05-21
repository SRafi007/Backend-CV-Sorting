@router.post(
    "/one-step", 
    response_model=List[MatchResponse],
    summary="Process job and resumes in one step"
)
async def process_one_step(
    background_tasks: BackgroundTasks,
    job: JobDesc = Depends(),
    files: List[UploadFile] = File(...),
    wait_for_results: bool = Query(False, description="Whether to wait for processing to complete")
):
    """
    Process job description and resumes in a single API call.
    
    - **job**: Job description details
    - **files**: List of resume files to analyze
    - **wait_for_results**: If true, wait for processing to complete and return results
    
    Returns either processing status or matching results depending on wait_for_results.
    """
    settings = get_settings()
    
    # Create a new session
    session_id = f"session_{uuid.uuid4().hex[:10]}"
    session_dir = create_session_dir(session_id)
    
    job_sessions[session_id] = {
        "created_at": datetime.now(),
        "status": "created",
        "files": [],
        "job": job
    }
    
    # Save uploaded files
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
    
    # Update session with files
    job_sessions[session_id]["files"] = uploaded_files
    job_sessions[session_id]["status"] = "processing"
    
    # Process the files
    if wait_for_results:
        # Process synchronously if waiting for results
        await process_session_files(session_id)
        if job_sessions[session_id]["status"] == "complete":
            return job_sessions[session_id].get("results", [])
        else:
            error_msg = job_sessions[session_id].get("error_message", "Unknown error during processing")
            raise HTTPException(status_code=500, detail=error_msg)
    else:
        # Process asynchronously if not waiting
        background_tasks.add_task(
            process_session_files,
            session_id=session_id
        )
        
        return {
            "job_id": session_id,
            "status": "processing",
            "total_resumes": len(uploaded_files),
            "estimated_completion": datetime.now() + timedelta(minutes=1),
            "message": "Processing started. Use GET /match/sessions/{job_id}/results to retrieve results."
        }