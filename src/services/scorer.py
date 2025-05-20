"""
Resume Scorer Service for calculating final match scores
"""
import logging
from typing import Tuple, List, Dict, Optional
import re
import spacy

from src.api.v1.schemas import JobDesc, EducationLevel
from src.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# NLP model
nlp = spacy.load("en_core_web_sm")


class ResumeScorer:
    """Service for scoring resume-job matches in detail"""
    
    def __init__(self):
        """Initialize scorer with config"""
        self.settings = get_settings()
        self.gpa_pattern = re.compile(r'(?:gpa|cgpa)[:\s]*([0-4][\.][0-9]{1,2})', re.IGNORECASE)

    def compute_final_score(
        self,
        similarity: float,
        resume_text: str,
        job: JobDesc,
        skills_matched: List[str],
        experience_years: Optional[float] = None,
        education_level: Optional[EducationLevel] = None,
        cgpa: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute final score based on multiple criteria
        
        Args:
            similarity: Content similarity score
            resume_text: Resume text content
            job: Job description
            skills_matched: List of matched skills
            experience_years: Years of experience detected
            education_level: Education level detected
            cgpa: CGPA/GPA detected
            
        Returns:
            Dictionary of scores by category and final composite score
        """
        # Initialize scores dictionary
        scores = {
            "similarity": similarity,
            "skills": 0.0,
            "experience": 0.0,
            "education": 0.0,
            "final": 0.0
        }
        
        # Calculate skills score
        if job.required_skills:
            required_matched = sum(1 for skill in job.required_skills 
                                   if skill.lower() in [s.lower() for s in skills_matched])
            scores["skills"] = required_matched / len(job.required_skills) if job.required_skills else 0.0
        else:
            # No specific skills required, use similarity as a proxy
            scores["skills"] = similarity
        
        # Calculate experience score
        if job.min_experience_years is not None and experience_years is not None:
            if experience_years >= job.min_experience_years:
                scores["experience"] = 1.0
            else:
                scores["experience"] = experience_years / job.min_experience_years
        else:
            # No specific experience requirement or couldn't detect experience
            scores["experience"] = 0.5
        
        # Calculate education score
        if job.min_cgpa is not None and cgpa is not None:
            if cgpa >= job.min_cgpa:
                scores["education"] = 1.0
            else:
                # Partial credit for close CGPA
                scores["education"] = cgpa / job.min_cgpa
        else:
            # No specific CGPA requirement or couldn't detect CGPA
            scores["education"] = 0.5
        
        # Calculate final weighted score
        weights = {
            "similarity": self.settings.SIMILARITY_WEIGHT,
            "skills": self.settings.SKILLS_WEIGHT,
            "experience": self.settings.EXPERIENCE_WEIGHT,
            "education": self.settings.EDUCATION_WEIGHT
        }
        
        scores["final"] = sum(scores[k] * weights[k] for k in weights)
        
        # Round all scores to 4 decimal places
        for key in scores:
            scores[key] = round(scores[key], 4)
        
        return scores

    def extract_cgpa(self, text: str) -> Optional[float]:
        """
        Extract CGPA/GPA from text
        
        Args:
            text: Text to extract CGPA from
            
        Returns:
            CGPA as float or None if not found
        """
        match = self.gpa_pattern.search(text)
        if match:
            try:
                cgpa = float(match.group(1))
                if 0.0 <= cgpa <= 4.0:
                    return cgpa
            except ValueError:
                pass
        return None

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract potential skills from job description text
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of potential skills
        """
        # Extract skills using spaCy
        doc = nlp(text)
        
        # Extract noun phrases as potential skills
        skills = []
        
        # Pattern-based extraction for common skill phrasings
        skill_patterns = [
            r'experience with ([\w\s]+)',
            r'knowledge of ([\w\s]+)',
            r'proficient in ([\w\s]+)',
            r'expertise in ([\w\s]+)',
            r'familiarity with ([\w\s]+)',
            r'skills? in ([\w\s]+)',
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).strip().rstrip('.,;')
                if len(skill) > 2 and not skill.lower().startswith(('a ', 'the ', 'an ')):
                    skills.append(skill.lower())
        
        # Use NLP to extract additional skills
        for chunk in doc.noun_chunks:
            # Filter to keep only technical terms
            if len(chunk.text) > 2 and not chunk.text.lower().startswith(('a ', 'the ', 'an ')):
                skills.append(chunk.text.lower())
        
        # Remove duplicates and return
        return list(set(skills))

    def match_resume_skills(self, resume_text: str, skill_list: List[str]) -> Tuple[float, List[str]]:
        """
        Match skills in resume text against a list of skills
        
        Args:
            resume_text: Resume text content
            skill_list: List of skills to match against
            
        Returns:
            Tuple of (match score, list of matched skills)
        """
        matched_skills = []
        
        for skill in skill_list:
            # Check for exact matches first
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, resume_text, re.IGNORECASE):
                matched_skills.append(skill)
                continue
                
            # Also check for variations
            # This is a simplified approach - a real implementation would be more sophisticated
            variations = [
                skill,  # original
                skill.replace(' ', '-'),  # hyphenated
                skill.replace(' ', ''),   # no spaces
                skill.replace('-', ' ')   # spaces instead of hyphens
            ]
            
            for var in variations:
                if var != skill:  # Skip the original which was already checked
                    pattern = r'\b' + re.escape(var) + r'\b'
                    if re.search(pattern, resume_text, re.IGNORECASE):
                        matched_skills.append(skill)  # Add the original skill name
                        break
        
        # Calculate match score
        if not skill_list:
            return 0.0, []
        
        match_score = len(matched_skills) / len(skill_list)
        return match_score, matched_skills