from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(job, resume_text: str) -> float:
    jd_emb = model.encode(job.description, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    return float(util.cos_sim(jd_emb, res_emb)[0][0])
