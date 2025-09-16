from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.forecasting_service import forecasting_service
from pydantic_models import InsightResponse
from db.redis_client import cache_insight, get_cached_insight_from_redis
import hashlib
import json

router = APIRouter(prefix="/predictive/insights", tags=["Predictive Layer (Insights)"])

class InsightRequest(BaseModel):
    shop_id: int
    item_id: int
    months: int = 3

@router.post("/explain", response_model=InsightResponse)
async def explain_forecast(request: InsightRequest):
    """
    Provides interpretability insights for a forecast (e.g., LSTM reliance, attention weights).
    Uses cached results if available, otherwise calls forecasting service.
    """
    # Generate cache key
    cache_key = hashlib.md5(json.dumps(request.dict(), sort_keys=True).encode()).hexdigest()
    cached = await get_cached_insight_from_redis(cache_key)
    if cached:
        return InsightResponse(**cached)
    
    try:
        # Generate forecast with insights
        result = await forecasting_service.generate_forecast_with_insights(
            request.shop_id, request.item_id, request.months
        )
        insights = result['insights'][0]  # Take first month's insights for simplicity
        
        # Placeholder for Ollama narrative (to be implemented)
        narrative = (
            f"The forecast for shop {insights['shop_name']} (ID: {request.shop_id}) and item {request.item_id} "
            f"relies {insights['lstm_trend_reliance']:.2%} on LSTM trends. "
            f"Attention weights indicate focus on recent data: t-2 ({insights['attention_t_minus_2']:.2f}), "
            f"t-1 ({insights['attention_t_minus_1']:.2f}), t-0 ({insights['attention_t_minus_0']:.2f})."
        )
        
        response = InsightResponse(
            forecast_id=cache_key,
            reliance_score=insights['lstm_trend_reliance'],
            attention_weights=[
                insights['attention_t_minus_2'],
                insights['attention_t_minus_1'],
                insights['attention_t_minus_0']
            ],
            narrative=narrative
        )
        
        # Cache the result
        await cache_insight(cache_key, response.dict(), ttl=86400)  # Cache for 24 hours
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")