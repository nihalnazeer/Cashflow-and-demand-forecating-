# api/analysis/__init__.py
from .historical import router as historical_router

__all__ = ["historical_router"]