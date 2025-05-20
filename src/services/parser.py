"""
Resume Parser Service for extracting information from resume files
"""
import logging
import re
import fitz  # PyMuPDF
import docx2txt
import spacy
from typing import Dict, List, Tuple, Optional, Union, BinaryIO
import asyncio
from io import BytesIO
import os

from src.api.v1.schemas import ParsedResume, EducationInfo, EducationLevel

# Configure logging
logger = logging.getLogger(__name__)

# Load NLP model if available (or we'll download it)
try:
    nlp = spacy.load("en_core_web_md")
except:
    logger.info("Downloading en_core_web_md model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")


class ResumeParser:
    """Service for parsing and extracting information from resumes"""
    
    # Common section headers in resumes
    SECTION_HEADERS = {
        'education': ['education', 'academic background', 'academic history', 'qualifications'],
        'experience': ['experience', 'work history', 'employment', 'work experience', 'professional experience'],
        'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise'],
        'projects': ['projects', 'project experience', 'academic projects', 'personal projects'],
        'certifications': ['certifications', 'certificates', 'professional certifications'],
        'publications': ['publications', 'research', 'papers', 'articles'],
        'summary': ['summary', 'professional summary', 'profile', 'about me'],
        'contact': ['contact', 'contact information', 'personal information'],
    }
    
    # Skills dictionary for common tech domains
    # This would be much more extensive in a real implementation
    SKILL_DICT = {
        'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'scala', 'kotlin', 'golang', 'swift'],
        'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'sklearn', 'data mining'],
        'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'redis', 'elasticsearch'],
        'web_dev': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'html', 'css', 'sass'],
        'devops': ['docker', 'kubernetes', 'aws', 'gcp', 'azure', 'jenkins', 'terraform', 'ansible'],
        'tools': ['git', 'jira', 'confluence', 'figma', 'adobe', 'photoshop', 'illustrator'],
    }
    
    # Education degrees and their levels
    EDUCATION_DEGREES = {
        'high_school': ['high school', 'secondary', 'ged'],
        'associate': ['associate', 'a.s.', 'a.a.'],
        'bachelor': ['bachelor', 'b.a.', 'b.s.', 'b.sc', 'b.e.', 'b.tech', 'undergraduate'],
        'master': ['master', 'm.a.', 'm.s.', 'm.sc', 'm.tech', 'mba', 'postgraduate'],
        'phd': ['phd', 'ph.d', 'doctorate', 'doctoral', 'd.phil'],
    }
    
    def __init__(self):
        self.gpa_pattern = re.compile(r'(?:gpa|cgpa)[:\s]*([0-4][\.][0-9]{1,2})', re.IGNORECASE)
        self.year_pattern = re.compile(r'(19|20)[0-9]{2}')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    
    async def parse_resume(self, file_content: bytes, filename: str) -> ParsedResume:
        """
        Parse a resume file and extract relevant information
        
        Args:
            file_content: Raw bytes of the resume file
            filename: Name of the file
            
        Returns:
            ParsedResume object with extracted information
        """
        try:
            # Extract text based on file extension
            ext = filename.split('.')[-1].lower()
            
            if ext == 'pdf':
                text = await self._extract_text_from_pdf(file_content)
            elif ext in ['docx', 'doc']:
                text = await self._extract_text_from_docx(file_content)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Basic parsed resume with text
            parsed = ParsedResume(
                filename=filename,
                text=text,
                skills=[],
                education=None,
                experience_years=None,
                extracted_sections={}
            )
            
            # Extract sections and information
            await asyncio.gather(
                self._extract_sections(parsed),
                self._extract_skills(parsed),
                self._extract_education(parsed),
                self._extract_experience(parsed)
            )
            
            return parsed
            
        except Exception as e:
            logger.exception(f"Error parsing resume {filename}: {str(e)}")
            # Return a basic parsed resume with just the text if we encounter errors
            return ParsedResume(
                filename=filename,
                text="Error parsing resume",
                skills=[],
                education=None,
                experience_years=None,
                extracted_sections={}
            )
    
    async def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    async def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            # Create a BytesIO object from the file content
            with BytesIO(file_content) as f:
                text = docx2txt.process(f)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise
    
    async def _extract_sections(self, parsed: ParsedResume) -> None:
        """
        Extract different sections from the resume
        
        Args:
            parsed: Partially filled ParsedResume object
        """
        text = parsed.text
        lines = text.split('\n')
        
        current_section = 'header'
        sections = {current_section: []}
        
        # First pass: identify section headers
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            found_section = None
            for section, headers in self.SECTION_HEADERS.items():
                if any(header.lower() in line.lower() for header in headers):
                    found_section = section
                    break
                    
            if found_section:
                current_section = found_section
                sections[current_section] = []
            else:
                sections[current_section].append(line)
        
        # Convert lists of lines to text blocks
        for section, lines in sections.items():
            parsed.extracted_sections[section] = '\n'.join(lines)
    
    async def _extract_skills(self, parsed: ParsedResume) -> None:
        """
        Extract skills from the resume
        
        Args:
            parsed: Partially filled ParsedResume object
        """
        # First check if there's a skills section
        skills_text = parsed.extracted_sections.get('skills', '')
        
        # If no dedicated skills section, use the whole text
        if not skills_text:
            skills_text = parsed.text
        
        # Extract skills from the flattened skill dictionary
        all_skills = []
        for category, skills in self.SKILL_DICT.items():
            for skill in skills:
                # Look for the skill as a word boundary
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, skills_text.lower()):
                    all_skills.append(skill)
        
        # Use NLP to find additional skills
        doc = nlp(parsed.text)
        
        # Look for noun phrases that might be skills
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Add any technical terms not in our dictionary
        # In a real implementation, this would use a more sophisticated approach
        
        # Remove duplicates and assign to parsed object
        parsed.skills = list(set(all_skills))
    
    async def _extract_education(self, parsed: ParsedResume) -> None:
        """
        Extract education information from the resume
        
        Args:
            parsed: Partially filled ParsedResume object
        """
        # Use education section if available
        edu_text = parsed.extracted_sections.get('education', '')
        
        # If no education section found, use the full text
        if not edu_text:
            edu_text = parsed.text
            
        # Create education info object
        education = EducationInfo()
        
        # Extract degree level
        for level, terms in self.EDUCATION_DEGREES.items():
            if any(term in edu_text.lower() for term in terms):
                education.level = level
                break
        
        # Extract CGPA
        cgpa_match = self.gpa_pattern.search(edu_text)
        if cgpa_match:
            try:
                education.cgpa = float(cgpa_match.group(1))
            except ValueError:
                pass
        
        # Extract graduation year
        year_matches = self.year_pattern.findall(edu_text)
        if year_matches:
            try:
                # Assume the largest year is the graduation year
                education.graduation_year = max(int(y) for y in year_matches)
            except ValueError:
                pass
        
        # Extract institution and major using NLP
        doc = nlp(edu_text)
        
        # Look for organization entities that might be universities
        orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        if orgs:
            education.institution = orgs[0]  # Take the first org as the institution
        
        # For major, look for specific keywords
        major_keywords = ['major', 'degree', 'in', 'program']
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(kw in sent_text for kw in major_keywords):
                # This is a simplistic approach - a real implementation would be more sophisticated
                for chunk in sent.noun_chunks:
                    if not education.major and any(term not in chunk.text.lower() for term in ['university', 'college', 'school']):
                        education.major = chunk.text
                        break
        
        # Assign to parsed object if we found anything useful
        if education.level or education.cgpa or education.institution or education.major:
            parsed.education = education
    
    async def _extract_experience(self, parsed: ParsedResume) -> None:
        """
        Extract work experience information from the resume
        
        Args:
            parsed: Partially filled ParsedResume object
        """
        # Use experience section if available
        exp_text = parsed.extracted_sections.get('experience', '')
        
        # If no experience section found, use the full text
        if not exp_text:
            exp_text = parsed.text
            
        # Extract years using regex
        years = self.year_pattern.findall(exp_text)
        
        if len(years) >= 2:
            try:
                # Convert to integers
                years = [int(y) for y in years]
                # Sort years
                years.sort()
                # Calculate experience (difference between min and max year)
                # This is a simplified approach - a real implementation would be more sophisticated
                experience_years = max(years) - min(years)
                
                # Cap at reasonable value and assign
                if 0 <= experience_years <= 40:
                    parsed.experience_years = float(experience_years)
            except ValueError:
                pass