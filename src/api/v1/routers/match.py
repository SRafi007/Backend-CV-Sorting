from fastapi import APIRouter, UploadFile, File, Depends
from typing import List
from src.api.v1.schemas import JobDesc, MatchResponse
from src.services.parser import parse_resume_text
from src.services.matcher import compute_similarity
from src.services.scorer import compute_final_score

router = APIRouter()


@router.post("/match", response_model=List[MatchResponse])
async def match_resumes(job: JobDesc, files: List[UploadFile] = File(...)):
    results = []
    # Extract job description details
    for file in files:
        text = await parse_resume_text(file)
        sim = compute_similarity(job, text)
        score, matched_skills, cgpa = compute_final_score(job, text, sim)
        results.append(
            {
                "resume": file.filename,
                "score": score,
                "similarity": sim,
                "cgpa": cgpa,
                "skills_matched": matched_skills,
            }
        )
    # sort results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
