# api/__init__.py
# Empty file or simple exports for modularity
from .analysis import historical
from .predictive import forecasts, insights

__all__ = ["historical", "forecasts", "insights"]