from src.utils.text_cleaner import extract_cgpa, extract_skills_from_text


def compute_final_score(job, resume_text: str, similarity: float):
    # CGPA
    cgpa = extract_cgpa(resume_text)
    cgpa_score = (cgpa / 4.0) if cgpa else 0.5

    # Skills
    req_skills = job.required_skills or extract_skills_from_text(job.description)
    matched = [s for s in req_skills if s.lower() in resume_text.lower()]
    skill_score = len(matched) / len(req_skills) if req_skills else 0

    # Combine
    final_score = 0.6 * similarity + 0.2 * skill_score + 0.2 * cgpa_score
    return round(final_score, 4), matched, cgpa
