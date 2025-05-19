from pydantic import BaseModel, Field
from typing import List, Optional


class JobDesc(BaseModel):
    title: str = Field(..., example="Data Scientist")
    description: str = Field(..., example="We need Python, ML, Pandas…")
    required_skills: List[str] = Field(default_factory=list)
    optional_skills: List[str] = Field(default_factory=list)
    min_experience_years: Optional[int] = Field(None, ge=0)
    min_cgpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    location: Optional[str] = Field(None)


class MatchResponse(BaseModel):
    resume: str
    score: float
    similarity: float
    cgpa: Optional[float]
    skills_matched: List[str]
