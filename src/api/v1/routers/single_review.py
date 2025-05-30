from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
from src.services.parser import ResumeParser
from src.services.matcher import ResumeMatcher
from src.api.v1.schemas import SingleReviewRequest, SingleReviewResponse, JobDesc
import logging

router = APIRouter(prefix="/single-review", tags=["Single Resume Review"])

logger = logging.getLogger(__name__)


@router.post("/", response_model=SingleReviewResponse)
async def review_single_resume(
    request: SingleReviewRequest, resume_file: UploadFile = File(...)
):
    # Step 1: Validate extension
    ext = resume_file.filename.split(".")[-1].lower()
    if ext not in {"pdf", "docx"}:
        raise HTTPException(
            status_code=400, detail="Only PDF and DOCX files are supported"
        )

    content = await resume_file.read()

    # Step 2: Parse resume
    parser = ResumeParser()
    parsed_resume = await parser.parse_resume(content, resume_file.filename)

    # Step 3: Create JobDesc object
    job = JobDesc(
        title=request.job_title,
        description=request.job_description,
        required_skills=request.required_skills or [],
        optional_skills=request.optional_skills or [],
        min_experience_years=request.min_experience_years,
        min_education=request.min_education,
    )

    # Step 4: Match
    matcher = ResumeMatcher()
    result = await matcher._score_resume(
        parsed_resume,
        job,
        await matcher._get_text_embedding(f"{job.title}. {job.description}"),
    )

    return result
