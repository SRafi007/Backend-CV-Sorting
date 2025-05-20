"""
Configuration module for the Resume Sorter API
"""
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "Resume Sorter API"
    APP_VERSION: str = "1.0.0"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
    # CORS settings
    #CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    CORS_ORIGINS: List[str] = ["*"]
    # NLP model settings
    NLP_MODEL: str = "all-MiniLM-L6-v2"
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx"]
    
    # Scoring weights
    SIMILARITY_WEIGHT: float = 0.5
    SKILLS_WEIGHT: float = 0.3
    EXPERIENCE_WEIGHT: float = 0.1
    EDUCATION_WEIGHT: float = 0.1
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Returns application settings as a singleton.
    Uses lru_cache to prevent re-loading settings on each call.
    """
    return Settings()