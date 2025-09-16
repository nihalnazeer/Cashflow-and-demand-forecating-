import asyncpg
from contextlib import asynccontextmanager
from config.settings import settings
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

pool = None

async def init_pool():
    global pool
    pool = await asyncpg.create_pool(settings.database_url)

@asynccontextmanager
async def get_db():
    async with pool.acquire() as conn:
        yield conn

async def query_historical_trends(db):
    """
    Legacy query for monthly sales trends (for compatibility).
    """
    query = """
        SELECT DATE_TRUNC('month', sale_date)::DATE AS month,
               SUM(total_daily_sales) AS total_revenue
        FROM daily_sales_summary
        GROUP BY month
        ORDER BY month
    """
    results = await db.fetch(query)
    return [dict(row) for row in results]

async def get_dynamic_query_results(dimension: str, metric: str, time_period: str, filters: dict = None, db=None):
    """
    Dynamically builds and executes a SQL query against historical_monthly_sales.
    Supports user-defined dimensions, metrics, time periods, and filters.
    """
    # Input Validation
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
        raise ValueError(f"Invalid dimension: {dimension}")
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}")

    # Query Building
    select_clause = f"{valid_dimensions[dimension]} AS {dimension}"
    metric_clause = f"{valid_metrics[metric]} AS {metric}"
    
    base_query = f"SELECT {select_clause}, {metric_clause} FROM historical_monthly_sales"
    
    where_clauses = []
    params = {}

    # Time Period Filter
    if time_period == "last_quarter":
        start_date = datetime.now() - timedelta(days=90)
        where_clauses.append("date >= $1")
        params["start_date"] = start_date
    elif time_period == "last_month":
        start_date = datetime.now() - timedelta(days=30)
        where_clauses.append("date >= $1")
        params["start_date"] = start_date
    elif time_period == "year_to_date":
        start_date = datetime(datetime.now().year, 1, 1)
        where_clauses.append("date >= $1")
        params["start_date"] = start_date
    else:
        raise ValueError(f"Invalid time_period: {time_period}")

    # Additional Filters
    if filters:
        param_index = len(params) + 1
        for key, value in filters.items():
            if key in valid_dimensions and isinstance(value, list):
                where_clauses.append(f"{key} = ANY(${param_index}::text[])")
                params[f"param_{param_index}"] = value
                param_index += 1

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
        
    group_by_clause = f"GROUP BY {dimension}"
    order_by_clause = f"ORDER BY {metric} DESC"
    full_query = f"{base_query} {group_by_clause} {order_by_clause} LIMIT 20"

    try:
        results = await db.fetch(full_query, *params.values())
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Dynamic query failed: {str(e)}")
        raise