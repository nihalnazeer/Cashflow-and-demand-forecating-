from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime

class HistoricalTrendResponse(BaseModel):
    month: datetime
    total_revenue: float

class Prediction(BaseModel):
    shop_id: int
    shop_name: str
    item_id: int
    item_category_name: str
    predicted_product_demand: int
    predicted_total_sales: float

class ForecastResponse(BaseModel):
    shop_id: int
    item_id: int
    months: int
    predictions: List[Prediction]
    confidence: float

class BatchForecastResponse(BaseModel):
    batch_results: List[ForecastResponse]

class InsightResponse(BaseModel):
    forecast_id: str
    reliance_score: float
    attention_weights: List[float]
    narrative: Optional[str]

class DynamicHistoricalResponse(BaseModel):
    dimension: Union[str, datetime]
    metric: float