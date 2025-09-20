from fastapi import APIRouter, HTTPException, Query
from pydantic_models import DynamicHistoricalResponse, MonthlySalesTrend
from db.postgres_client import get_db
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis/historical", tags=["Historical Analysis"])

# --- Caching and Helper Function ---
_latest_data_date_cache: Optional[date] = None

async def get_latest_data_date() -> date:
    """
    Finds the latest date from the sales data.
    Uses a simple cache to avoid querying the database on every request.
    """
    global _latest_data_date_cache
    if _latest_data_date_cache:
        return _latest_data_date_cache
    
    query = "SELECT MAX(date)::DATE as max_date FROM historical_monthly_sales;"
    try:
        async with get_db() as db:
            result = await db.fetchrow(query)
        if result and result['max_date']:
            _latest_data_date_cache = result['max_date']
            logger.info(f"Latest data date cached: {_latest_data_date_cache}")
            return _latest_data_date_cache
        else:
            return datetime.now().date()
    except Exception as e:
        logger.error(f"Failed to get latest data date: {e}")
        return datetime.now().date()


# --- Helper Functions (Updated to use the dynamic date) ---

async def get_dynamic_query_results(dimension: str, metric: str, time_period: str, filters: Dict = None):
    valid_dimensions = { "shop_name": "shop_name", "item_category_name": "item_category_name", "month": "DATE_TRUNC('month', date)::DATE" }
    valid_metrics = { "total_revenue": "SUM(monthly_revenue)", "total_demand": "SUM(item_cnt_month)" }
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension: {dimension}")
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}")

    select_clause = f"{valid_dimensions[dimension]} AS dimension, {valid_metrics[metric]} AS metric"
    base_query = f"SELECT {select_clause} FROM historical_monthly_sales"
    where_clauses = []
    params = []

    latest_date = await get_latest_data_date()
    
    # ✨ FIX: Added 'all_time' handling for consistency
    if time_period == "last_quarter":
        start_date = latest_date - timedelta(days=90)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "last_month":
        start_date = latest_date - timedelta(days=30)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "year_to_date":
        start_date = datetime(latest_date.year, 1, 1).date()
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period != "all_time":
        raise ValueError(f"Invalid time_period: {time_period}. Choose from ['last_quarter', 'last_month', 'year_to_date', 'all_time']")
        
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
        async with get_db() as db:
            results = await db.fetch(base_query, *params)
        if not results:
            raise HTTPException(status_code=404, detail="No data found for the specified criteria")
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")


async def get_kpis(time_period: str, filters: Dict = None):
    base_query = "SELECT SUM(monthly_revenue) as total_revenue, SUM(item_cnt_month) as total_demand FROM historical_monthly_sales"
    where_clauses = []
    params = []
    
    latest_date = await get_latest_data_date()

    # ✨ FIX: Added 'all_time' handling for consistency
    if time_period == "last_quarter":
        start_date = latest_date - timedelta(days=90)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "last_month":
        start_date = latest_date - timedelta(days=30)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "year_to_date":
        start_date = datetime(latest_date.year, 1, 1).date()
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period != "all_time":
        raise ValueError(f"Invalid time_period: {time_period}. Choose from ['last_quarter', 'last_month', 'year_to_date', 'all_time']")
        
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
        async with get_db() as db:
            result = await db.fetchrow(base_query, *params)
        if not result or result['total_revenue'] is None:
            raise HTTPException(status_code=404, detail="No data found for specified criteria")
        return {
            "total_revenue": float(result['total_revenue']),
            "total_demand": float(result['total_demand'])
        }
    except Exception as e:
        logger.error(f"KPI query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch KPIs: {str(e)}")


# --- API Endpoints ---

@router.get("/trends/overall", response_model=List[MonthlySalesTrend])
async def get_overall_sales_trend():
    query = "SELECT DATE_TRUNC('month', date)::DATE AS month, SUM(monthly_revenue) AS total_revenue FROM historical_monthly_sales GROUP BY month ORDER BY month ASC"
    try:
        async with get_db() as db:
            results = await db.fetch(query)
        if not results:
            raise HTTPException(status_code=404, detail="No sales data found in the database.")
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Overall sales trend query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch overall sales trend.")


@router.get("/trends/detailed", response_model=List[MonthlySalesTrend])
async def get_detailed_sales_trend(
    time_period: str = "year_to_date",
    shop_names: Optional[List[str]] = Query(None, alias="shop_name"),
):
    base_query = "SELECT DATE_TRUNC('month', date)::DATE AS month, SUM(monthly_revenue) AS total_revenue FROM historical_monthly_sales"
    where_clauses = []
    params = []
    
    latest_date = await get_latest_data_date()

    if time_period == "last_quarter":
        start_date = latest_date - timedelta(days=90)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "last_month":
        start_date = latest_date - timedelta(days=30)
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period == "year_to_date":
        start_date = datetime(latest_date.year, 1, 1).date()
        where_clauses.append(f"date >= ${len(params) + 1}")
        params.append(start_date)
    elif time_period != "all_time":
        raise HTTPException(
            status_code=400,
            detail="Invalid time_period. Choose from ['last_month', 'last_quarter', 'year_to_date', 'all_time']"
        )
        
    if shop_names:
        where_clauses.append(f"shop_name = ANY(${len(params) + 1}::text[])")
        params.append(shop_names)

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    
    full_query = base_query + " GROUP BY month ORDER BY month ASC;"
    try:
        async with get_db() as db:
            results = await db.fetch(full_query, *params)
        if not results:
            raise HTTPException(status_code=404, detail="No data found for the specified filters.")
        return [dict(row) for row in results]
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Detailed sales trend query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch detailed sales trend: {str(e)}")


@router.get("/kpis", response_model=dict)
async def get_historical_kpis(
    time_period: str = "last_quarter",
    shop_names: Optional[List[str]] = Query(None, alias="shop_name"),
    item_categories: Optional[List[str]] = Query(None, alias="item_category"),
):
    filters = {}
    if shop_names:
        filters["shop_name"] = shop_names
    if item_categories:
        filters["item_category_name"] = item_categories
    return await get_kpis(time_period, filters)


@router.get("/trends/{dimension}", response_model=List[DynamicHistoricalResponse])
async def get_historical_trends(
    dimension: str,
    metric: str = "total_revenue",
    time_period: str = "last_quarter",
    shop_names: Optional[List[str]] = Query(None, alias="shop_name"),
    item_categories: Optional[List[str]] = Query(None, alias="item_category"),
):
    filters = {}
    if shop_names:
        filters["shop_name"] = shop_names
    if item_categories:
        filters["item_category_name"] = item_categories
    return await get_dynamic_query_results(dimension, metric, time_period, filters)


@router.get("/shops", response_model=List[str])
async def get_all_shop_names():
    query = "SELECT DISTINCT shop_name FROM historical_monthly_sales ORDER BY shop_name;"
    try:
        async with get_db() as db:
            results = await db.fetch(query)
        if not results:
            raise HTTPException(status_code=404, detail="No shop names found.")
        return [row['shop_name'] for row in results]
    except Exception as e:
        logger.error(f"Failed to fetch shop names: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch shop names.")