from fastapi import APIRouter, HTTPException, Depends
from pydantic_models import HistoricalTrendResponse, DynamicHistoricalResponse
from db.postgres_client import get_db
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis/historical", tags=["Historical Analysis"])

async def get_dynamic_query_results(dimension: str, metric: str, time_period: str, filters: Dict = None, db=Depends(get_db)):
    """
    Dynamically builds and executes an async SQL query against historical_monthly_sales.
    Returns aggregated data by dimension and metric.
    """
    valid_dimensions = {
        "shop_name": "shop_name",
        "item_category_name": "item_category_name",
        "month": "DATE_TRUNC('month', date)::DATE"
    }
    valid_metrics = {
        "total_revenue": "SUM(monthly_revenue)",
        "total_demand": "SUM(item_cnt_month)"
    }
    
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension: {dimension}. Choose from {list(valid_dimensions.keys())}")
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Choose from {list(valid_metrics.keys())}")

    select_clause = f"{valid_dimensions[dimension]} AS dimension, {valid_metrics[metric]} AS metric"
    base_query = f"SELECT {select_clause} FROM historical_monthly_sales"
    
    where_clauses = []
    params = []
    
    # Time Period Filter
    if time_period == "last_quarter":
        start_date = datetime.now() - timedelta(days=90)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "last_month":
        start_date = datetime.now() - timedelta(days=30)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "year_to_date":
        start_date = datetime(datetime.now().year, 1, 1)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    else:
        raise ValueError(f"Invalid time_period: {time_period}. Choose from ['last_quarter', 'last_month', 'year_to_date']")

    # Additional Filters
    if filters:
        valid_filter_keys = ["shop_name", "item_category_name"]
        for key, value in filters.items():
            if key in valid_filter_keys and isinstance(value, list):
                where_clauses.append(f"{key} = ANY(${len(params) + 1}::text[])")
                params.append(value)
            else:
                raise ValueError(f"Invalid filter key or value: {key}={value}")

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    
    base_query += f" GROUP BY {valid_dimensions[dimension]} ORDER BY metric DESC LIMIT 20"
    
    try:
        results = await db.fetch(base_query, *params)
        return [
            DynamicHistoricalResponse(
                dimension=row['dimension'],
                metric=float(row['metric'])
            ) for row in results
        ]
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

async def get_kpis(time_period: str, filters: Dict = None, db=Depends(get_db)):
    """
    Calculates KPIs (total revenue and demand) for a given time period with optional filters.
    Returns a single row of data.
    """
    base_query = "SELECT SUM(monthly_revenue) as total_revenue, SUM(item_cnt_month) as total_demand FROM historical_monthly_sales"
    
    where_clauses = []
    params = []
    
    # Time Period Filter
    if time_period == "last_quarter":
        start_date = datetime.now() - timedelta(days=90)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "last_month":
        start_date = datetime.now() - timedelta(days=30)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "year_to_date":
        start_date = datetime(datetime.now().year, 1, 1)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    else:
        raise ValueError(f"Invalid time_period: {time_period}")

    # Additional Filters
    if filters:
        valid_filter_keys = ["shop_name", "item_category_name"]
        for key, value in filters.items():
            if key in valid_filter_keys and isinstance(value, list):
                where_clauses.append(f"{key} = ANY(${len(params) + 1}::text[])")
                params.append(value)
            else:
                raise ValueError(f"Invalid filter key or value: {key}={value}")

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    
    try:
        result = await db.fetch_one(base_query, *params)
        if not result:
            raise HTTPException(status_code=404, detail="No data found for specified criteria")
        return {
            "total_revenue": float(result['total_revenue']) if result['total_revenue'] is not None else 0.0,
            "total_demand": float(result['total_demand']) if result['total_demand'] is not None else 0.0
        }
    except Exception as e:
        logger.error(f"KPI query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch KPIs: {str(e)}")

@router.get("/trends/{dimension}", response_model=List[DynamicHistoricalResponse])
async def get_historical_trends(
    dimension: str,
    metric: str = "total_revenue",
    time_period: str = "last_quarter",
    shop_names: Optional[List[str]] = None,
    item_categories: Optional[List[str]] = None,
    db=Depends(get_db)
):
    """
    Retrieves historical trends by dimension (e.g., shop_name, item_category_name, month).
    Supports filtering by shop_names or item_categories.
    """
    filters = {}
    if shop_names:
        filters["shop_name"] = shop_names
    if item_categories:
        filters["item_category_name"] = item_categories
    
    return await get_dynamic_query_results(dimension, metric, time_period, filters, db)

@router.get("/kpis", response_model=dict)
async def get_historical_kpis(
    time_period: str = "last_quarter",
    shop_names: Optional[List[str]] = None,
    item_categories: Optional[List[str]] = None,
    db=Depends(get_db)
):
    """
    Retrieves KPIs (total revenue and demand) for a given time period.
    Supports filtering by shop_names or item_categories.
    """
    filters = {}
    if shop_names:
        filters["shop_name"] = shop_names
    if item_categories:
        filters["item_category_name"] = item_categories
    
    return await get_kpis(time_period, filters, db)