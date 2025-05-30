"""
Resume Sorter API - Main Application Entry Point
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.v1.routers import match_router
from src.api.v1.routers import single_review

from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Load models, establish connections
    logger.info("Starting Resume Sorter API")
    # Start any background services here
    yield
    # Shutdown: Release resources
    logger.info("Shutting down Resume Sorter API")
    # Clean up resources here


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title="Resume Sorter API",
        description="API for ranking and filtering resumes based on job descriptions",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(match_router, prefix="/api/v1")
    app.include_router(single_review, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
