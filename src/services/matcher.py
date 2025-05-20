"""
Resume Matcher Service for scoring and ranking resumes against job descriptions
"""
import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Any
import asyncio
from sentence_transformers import SentenceTransformer, util
import spacy
import re
from functools import lru_cache

from src.api.v1.schemas import JobDesc, ParsedResume, MatchResponse, SkillMatch, EducationLevel
from src.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_md")


class ResumeMatcher:
    """Service for matching and ranking resumes against job descriptions"""
    
    def __init__(self):
        """Initialize the matcher with embedding model"""
        self.settings = get_settings()
        # Load the embedding model
        try:
            self.model = SentenceTransformer(self.settings.NLP_MODEL)
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
            
        # Define scoring weights
        self.weights = {
            "similarity": self.settings.SIMILARITY_WEIGHT,
            "skills": self.settings.SKILLS_WEIGHT,
            "experience": self.settings.EXPERIENCE_WEIGHT,
            "education": self.settings.EDUCATION_WEIGHT
        }
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
    
    async def rank_resumes(self, job: JobDesc, parsed_resumes: List[ParsedResume]) -> List[MatchResponse]:
        """
        Score and rank resumes against a job description
        
        Args:
            job: Job description with requirements
            parsed_resumes: List of parsed resume objects
            
        Returns:
            List of MatchResponse objects sorted by score
        """
        # Extract key information from job description
        job_text = f"{job.title}. {job.description}"
        
        # Create job embedding asynchronously
        job_embedding = await self._get_text_embedding(job_text)
        
        # Process each resume in parallel
        tasks = [
            self._score_resume(resume, job, job_embedding) 
            for resume in parsed_resumes
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sort by overall score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    @lru_cache(maxsize=100)
    async def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using SentenceTransformer
        Cache results to avoid recomputing the same embeddings
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_tensor=True)
    
    async def _score_resume(
        self, 
        resume: ParsedResume, 
        job: JobDesc, 
        job_embedding: Any
    ) -> MatchResponse:
        """
        Score a single resume against job requirements
        
        Args:
            resume: Parsed resume object
            job: Job description
            job_embedding: Precomputed job description embedding
            
        Returns:
            MatchResponse with scores and match details
        """
        # Get resume text embedding
        resume_embedding = await self._get_text_embedding(resume.text)
        
        # Calculate semantic similarity
        similarity = float(util.cos_sim(job_embedding, resume_embedding)[0][0])
        
        # Calculate skills match
        skills_score, skills_matched, skills_missing = await self._score_skills(resume, job)
        
        # Calculate experience match
        experience_score = await self._score_experience(resume, job)
        
        # Calculate education match
        education_score = await self._score_education(resume, job)
        
        # Calculate weighted total score
        total_score = (
            self.weights["similarity"] * similarity +
            self.weights["skills"] * skills_score +
            self.weights["experience"] * experience_score +
            self.weights["education"] * education_score
        )
        
        # Round to 4 decimal places
        total_score = round(total_score, 4)
        
        # Create match response
        return MatchResponse(
            resume_id=resume.filename,
            score=total_score,
            similarity=similarity,
            education=resume.education,
            experience_years=resume.experience_years,
            skills_matched=skills_matched,
            skills_missing=skills_missing,
            match_details={
                "content_similarity": round(similarity, 4),
                "skills_match": round(skills_score, 4),
                "experience_match": round(experience_score, 4),
                "education_match": round(education_score, 4)
            }
        )
    
    async def _score_skills(
        self, 
        resume: ParsedResume, 
        job: JobDesc
    ) -> Tuple[float, List[SkillMatch], List[str]]:
        """
        Score resume based on skills match
        
        Args:
            resume: Parsed resume
            job: Job description
            
        Returns:
            Tuple of (score, matched_skills, missing_skills)
        """
        # Extract skills from job if not provided
        required_skills = job.required_skills
        if not required_skills and job.description:
            # Extract skills from description using NLP
            # This is a simplified approach - a real implementation would be more sophisticated
            required_skills = await self._extract_skills_from_text(job.description)
        
        optional_skills = job.optional_skills or []
        
        # Convert resume skills to lowercase set for case-insensitive matching
        resume_skills_set = {skill.lower() for skill in resume.skills}
        
        # Match required skills
        matched_required = []
        missing_required = []
        
        for skill in required_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_skills_set:
                # Find context for this skill in the resume text
                context = self._extract_skill_context(resume.text, skill)
                matched_required.append(SkillMatch(
                    skill=skill,
                    required=True,
                    context=context
                ))
            else:
                missing_required.append(skill)
        
        # Match optional skills
        matched_optional = []
        for skill in optional_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_skills_set:
                context = self._extract_skill_context(resume.text, skill)
                matched_optional.append(SkillMatch(
                    skill=skill, 
                    required=False,
                    context=context
                ))
        
        # All matched skills
        all_matched = matched_required + matched_optional
        
        # Calculate score
        if not required_skills:
            # If no required skills specified, calculate based on optional
            if not optional_skills:
                # No skills specified at all
                return 0.5, all_matched, missing_required
            else:
                # Only optional skills
                match_ratio = len(matched_optional) / len(optional_skills) if optional_skills else 0
                return match_ratio, all_matched, missing_required
        else:
            # Required skills are specified
            required_ratio = len(matched_required) / len(required_skills)
            
            # Optional skills provide a bonus
            optional_bonus = 0
            if optional_skills:
                optional_bonus = 0.2 * (len(matched_optional) / len(optional_skills))
            
            # Combine scores with a cap at 1.0
            return min(1.0, required_ratio + optional_bonus), all_matched, missing_required
    
    def _extract_skill_context(self, text: str, skill: str) -> str:
        """
        Extract the context around a skill mention in the text
        
        Args:
            text: Resume text
            skill: Skill to find context for
            
        Returns:
            String containing context or None if not found
        """
        # Find the skill in the text (case insensitive)
        pattern = re.compile(r'(?i)\b' + re.escape(skill) + r'\b')
        match = pattern.search(text)
        
        if not match:
            return None
            
        # Get indices
        start_idx = max(0, match.start() - 50)
        end_idx = min(len(text), match.end() + 50)
        
        # Extract context
        context = text[start_idx:end_idx]
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    async def _extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract potential skills from text using NLP
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of extracted skills
        """
        # This is a simplified implementation
        # A real-world solution would use a more sophisticated approach
        
        # Process with spaCy
        doc = nlp(text)
        
        # Extract noun phrases as potential skills
        skills = []
        for chunk in doc.noun_chunks:
            # Filter to keep only technical terms
            if len(chunk.text) > 2 and not chunk.text.lower().startswith(('a ', 'the ', 'an ')):
                skills.append(chunk.text.lower())
        
        # Add named entities that might be technologies
        for ent in doc.ents:
            if ent.label_ in ('PRODUCT', 'ORG', 'GPE'):
                if len(ent.text) > 2:
                    skills.append(ent.text.lower())
        
        # Remove duplicates and return
        return list(set(skills))
    
    async def _score_experience(self, resume: ParsedResume, job: JobDesc) -> float:
        """
        Score resume based on experience match
        
        Args:
            resume: Parsed resume
            job: Job description
            
        Returns:
            Experience match score (0.0 to 1.0)
        """
        # If job doesn't specify minimum experience, give full score
        if job.min_experience_years is None:
            return 1.0
            
        # If resume doesn't have detected experience, give minimum score
        if resume.experience_years is None:
            return 0.1
        
        # Calculate ratio of experience
        ratio = resume.experience_years / job.min_experience_years
        
        # Score with diminishing returns for excess experience
        if ratio >= 1.0:
            # Meets minimum requirement
            excess = ratio - 1.0
            # Cap at 1.5x minimum for full score
            return min(1.0, 0.8 + (0.2 * min(1.0, excess / 0.5)))
        else:
            # Below minimum requirement
            # Give partial credit
            return 0.1 + (0.7 * ratio)
    
    async def _score_education(self, resume: ParsedResume, job: JobDesc) -> float:
        """
        Score resume based on education match
        
        Args:
            resume: Parsed resume
            job: Job description
            
        Returns:
            Education match score (0.0 to 1.0)
        """
        # Education level ranking
        edu_levels = {
            EducationLevel.HIGH_SCHOOL: 1,
            EducationLevel.ASSOCIATE: 2,
            EducationLevel.BACHELOR: 3,
            EducationLevel.MASTER: 4,
            EducationLevel.PHD: 5
        }
        
        # If job doesn't specify minimum education, give full score
        if job.min_education is None:
            return 1.0
            
        # If resume doesn't have detected education, check for hints in text
        if (resume.education is None or resume.education.level is None):
            # Give a moderate score as default
            return 0.5
        
        # Get numeric values for comparison
        job_level = edu_levels.get(job.min_education, 0)
        resume_level = edu_levels.get(resume.education.level, 0)
        
        # Calculate education score
        if resume_level >= job_level:
            # Meets or exceeds minimum education
            return 1.0
        else:
            # Below minimum education
            diff = job_level - resume_level
            return max(0.1, 1.0 - (0.25 * diff))