"""
API Router initialization
"""
from fastapi import APIRouter

from src.api.v1.routers.match import router as match_router

__all__ = ["match_router"]