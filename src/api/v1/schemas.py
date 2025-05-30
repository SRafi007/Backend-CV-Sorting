"""
Pydantic schemas for the Resume Sorter API
"""

from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class EducationLevel(str, Enum):
    """Education level enum"""

    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


class JobDesc(BaseModel):
    """Job description schema"""

    title: str = Field(..., description="Job title", example="Senior Data Scientist")
    description: str = Field(
        ...,
        description="Full job description text",
        example="We're looking for an experienced Data Scientist...",
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="List of required skills",
        example=["Python", "Machine Learning", "SQL"],
    )
    optional_skills: List[str] = Field(
        default_factory=list,
        description="List of preferred but not required skills",
        example=["Docker", "AWS", "Spark"],
    )
    min_experience_years: Optional[int] = Field(
        None, ge=0, description="Minimum years of experience required", example=3
    )
    min_education: Optional[EducationLevel] = Field(
        None,
        description="Minimum education level required",
        example=EducationLevel.BACHELOR,
    )
    min_cgpa: Optional[float] = Field(
        None, ge=0.0, le=4.0, description="Minimum CGPA/GPA required", example=3.0
    )
    location: Optional[str] = Field(
        None, description="Job location", example="San Francisco, CA"
    )
    industry: Optional[str] = Field(
        None, description="Industry sector", example="Healthcare"
    )

    @field_validator("required_skills", "optional_skills")
    @classmethod
    def normalize_skills(cls, v):
        """Normalize skills to lowercase and remove duplicates"""
        if not v:
            return []
        return list(set(skill.lower().strip() for skill in v))


class SkillMatch(BaseModel):
    """Skill match details"""

    skill: str = Field(..., example="Python")
    required: bool = Field(..., example=True)
    context: Optional[str] = Field(
        None, example="5+ years of Python experience in data science"
    )


class EducationInfo(BaseModel):
    """Education information extracted from resume"""

    level: Optional[EducationLevel] = Field(None, example=EducationLevel.MASTER)
    institution: Optional[str] = Field(None, example="Stanford University")
    major: Optional[str] = Field(None, example="Computer Science")
    cgpa: Optional[float] = Field(None, ge=0.0, le=4.0, example=3.8)
    graduation_year: Optional[int] = Field(None, example=2020)


class MatchResponse(BaseModel):
    """Response schema for match endpoint"""

    resume_id: str = Field(..., example="resume_1.pdf")
    score: float = Field(..., ge=0.0, le=1.0, example=0.85)
    similarity: float = Field(..., ge=0.0, le=1.0, example=0.78)
    education: Optional[EducationInfo] = None
    experience_years: Optional[float] = Field(None, example=4.5)
    skills_matched: List[SkillMatch] = Field(default_factory=list)
    skills_missing: List[str] = Field(default_factory=list)
    match_details: Dict[str, float] = Field(
        ...,
        example={
            "content_similarity": 0.78,
            "skills_match": 0.90,
            "experience_match": 0.85,
            "education_match": 0.80,
        },
    )


class ParsedResume(BaseModel):
    """Internal model for parsed resume data"""

    filename: str
    text: str
    skills: List[str] = Field(default_factory=list)
    education: Optional[EducationInfo] = None
    experience_years: Optional[float] = None
    extracted_sections: Dict[str, str] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class BatchProcessResponse(BaseModel):
    """Response for batch processing request"""

    job_id: str = Field(..., example="job_12345")
    status: str = Field(..., example="processing")
    total_resumes: int = Field(..., example=25)
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str
    detail: Optional[str] = None
    code: int


class SingleReviewRequest(BaseModel):
    job_title: str
    job_description: str
    required_skills: Optional[List[str]] = []
    optional_skills: Optional[List[str]] = []
    min_experience_years: Optional[float] = None
    min_education: Optional[str] = None  # Enum or str


class SingleReviewResponse(BaseModel):
    resume_id: str
    score: float
    similarity: float
    education: Optional[dict]
    experience_years: Optional[float]
    skills_matched: List[dict]
    skills_missing: List[str]
    match_details: dict
