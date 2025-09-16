# api/predictive/forecasts.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from pydantic import BaseModel, Field
from core.forecasting_service import forecasting_service
from pydantic_models import ForecastResponse  # Assumed: e.g., {"shop_id": int, "item_id": int, "months": int, "predictions": List[float], "confidence": float}

router = APIRouter(prefix="/predictive/forecasts", tags=["Predictive Layer (Forecasts)"])

class ForecastRequest(BaseModel):
    shop_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    months: int = Field(3, ge=1, le=12)  # Limit to reasonable horizon

@router.post("/generate", response_model=ForecastResponse)
async def generate_on_demand_forecast(request: ForecastRequest):
    """
    Generates a live forecast for the next N months for a specific shop-item pair.
    Calls the HALSTM service for real-time inference.
    """
    try:
        predictions = forecasting_service.generate_forecast(
            request.shop_id, request.item_id, request.months
        )
        # Mock confidence (in prod, compute from ensemble or variance)
        confidence = 0.85  # Placeholder
        return ForecastResponse(
            shop_id=request.shop_id,
            item_id=request.item_id,
            months=request.months,
            predictions=predictions,
            confidence=confidence
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.get("/generate/{shop_id}/{item_id}/{months}")
async def get_forecast_get(shop_id: int, item_id: int, months: int = 3):
    """
    GET variant for simple URL-based requests (e.g., from dashboard links).
    Mirrors the POST but without body.
    """
    if months > 12:
        raise HTTPException(status_code=400, detail="Max 12 months allowed")
    return await generate_on_demand_forecast(ForecastRequest(shop_id=shop_id, item_id=item_id, months=months))

# Bulk endpoint for multiple items (e.g., category-level)
@router.post("/batch-generate")
async def generate_batch_forecasts(request: List[ForecastRequest]):
    """
    Batch forecasts for efficiency (e.g., all items in a shop).
    Processes in parallel if service supports it.
    """
    results = []
    for req in request:
        results.append(await generate_on_demand_forecast(req))
    return {"batch_results": results}