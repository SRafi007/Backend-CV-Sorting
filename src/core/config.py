from pydantic import BaseSettings, AnyHttpUrl
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Resume Sorter API"
    API_VERSION: str = "v1"
    CORS_ORIGINS: List[AnyHttpUrl] = ["*"]
    # add other settings like BUCKET_URL, DB_URL

    class Config:
        env_file = ".env"


settings = Settings()
