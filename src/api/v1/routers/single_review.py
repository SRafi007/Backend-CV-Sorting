from fastapi import APIRouter, UploadFile, File, HTTPException,Form
from typing import Dict
from src.services.parser import ResumeParser
from src.services.matcher import ResumeMatcher
from src.api.v1.schemas import SingleReviewRequest, SingleReviewResponse, JobDesc
import logging
from pydantic import BaseModel
from typing import Optional, List



router = APIRouter(prefix="/single-review", tags=["Single Resume Review"])

logger = logging.getLogger(__name__)


class SingleReviewRequest(BaseModel):
    job_title: str
    job_description: str
    required_skills: Optional[List[str]] = []
    optional_skills: Optional[List[str]] = []
    min_experience_years: Optional[float] = None
    min_education: Optional[str] = None


@router.post("/", response_model=SingleReviewResponse)
async def review_single_resume(
    job_title: str = Form(...),
    job_description: str = Form(...),
    required_skills: Optional[List[str]] = Form(default=[]),
    optional_skills: Optional[List[str]] = Form(default=[]),
    min_experience_years: Optional[float] = Form(default=None),
    min_education: Optional[str] = Form(default=None),
    resume_file: UploadFile = File(...)
):
    job = JobDesc(
        title=job_title,
        description=job_description,
        required_skills=required_skills,
        optional_skills=optional_skills,
        min_experience_years=min_experience_years,
        min_education=min_education,
    )

    content = await resume_file.read()
    parser = ResumeParser()
    parsed_resume = await parser.parse_resume(content, resume_file.filename)

    matcher = ResumeMatcher()
    result = await matcher._score_resume(parsed_resume, job, await matcher._get_text_embedding(f"{job.title}. {job.description}"))

    return result