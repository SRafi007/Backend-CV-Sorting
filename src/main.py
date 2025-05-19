from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.routers.match import router as match_router
from src.core.middleware import LoggingMiddleware
from src.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
)

# Register middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(match_router, prefix="/api/v1", tags=["match"])


# root endpoint
@app.get("/", tags=["root"])
async def read_root():
    return {"message": "Welcome to the Resume Sorter API!"}
