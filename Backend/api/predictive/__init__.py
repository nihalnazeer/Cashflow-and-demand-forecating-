# api/predictive/__init__.py
from .forecasts import router as forecasts_router
from .insights import router as insights_router

__all__ = ["forecasts_router", "insights_router"]