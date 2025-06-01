from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from src.api.v1.schemas import JobDesc, MatchResponse
from src.services.parser import ResumeParser
from src.services.matcher import ResumeMatcher
from src.services.scorer import ResumeScorer
import logging

router = APIRouter(prefix="/single-review", tags=["Single Resume Review"])
logger = logging.getLogger(__name__)


class SingleReviewRequest(BaseModel):
    job_title: str
    job_description: str
    required_skills: Optional[List[str]] = []
    optional_skills: Optional[List[str]] = []
    min_experience_years: Optional[float] = None
    min_education: Optional[str] = None
    min_cgpa: Optional[float] = None


@router.post("/", response_model=MatchResponse)
async def review_single_resume(
    job_title: str = Form(...),
    job_description: str = Form(...),
    required_skills: Optional[List[str]] = Form(default=[]),
    optional_skills: Optional[List[str]] = Form(default=[]),
    min_experience_years: Optional[float] = Form(default=None),
    min_education: Optional[str] = Form(default=None),
    min_cgpa: Optional[float] = Form(default=None),
    resume_file: UploadFile = File(...)
):
    # Step 1: Parse resume
    parser = ResumeParser()
    content = await resume_file.read()
    parsed_resume = await parser.parse_resume(content, resume_file.filename)

    # Step 2: Create job object
    job = JobDesc(
        title=job_title,
        description=job_description,
        required_skills=required_skills,
        optional_skills=optional_skills,
        min_experience_years=min_experience_years,
        min_education=min_education,
        min_cgpa=min_cgpa
    )

    # Step 3: Embedding + similarity
    matcher = ResumeMatcher()
    job_text = f"{job.title}. {job.description}"
    job_embedding = await matcher._get_text_embedding(job_text)
    resume_embedding = await matcher._get_text_embedding(parsed_resume.text)
    similarity = float(matcher.model.similarity_fct(job_embedding, resume_embedding).cpu().numpy()[0][0])

    # Step 4: Use scorer
    scorer = ResumeScorer()
    _, matched_skills = scorer.match_resume_skills(parsed_resume.text, required_skills + optional_skills)
    cgpa = parsed_resume.education.cgpa if parsed_resume.education else None
    final_scores = scorer.compute_final_score(
        similarity=similarity,
        resume_text=parsed_resume.text,
        job=job,
        skills_matched=matched_skills,
        experience_years=parsed_resume.experience_years,
        education_level=parsed_resume.education.level if parsed_resume.education else None,
        cgpa=cgpa
    )

    return MatchResponse(
        resume_id=parsed_resume.filename,
        score=final_scores["final"],
        similarity=similarity,
        education=parsed_resume.education,
        experience_years=parsed_resume.experience_years,
        skills_matched=[],  # You can populate with SkillMatch list if needed
        skills_missing=[],
        match_details=final_scores
    )
