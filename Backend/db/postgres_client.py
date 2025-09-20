# db/postgres_client.py
import asyncpg
from contextlib import asynccontextmanager
from config.settings import settings
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global pool reference
pool: Optional[asyncpg.pool.Pool] = None


async def init_pool():
    """
    Initialize the asyncpg connection pool.
    Should be called on FastAPI startup.
    """
    global pool
    if pool is None:
        try:
            pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL,  # ✅ FIXED: uppercase field
                min_size=1,
                max_size=10
            )
            logger.info("Postgres connection pool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Postgres pool: {str(e)}")
            raise


async def close_pool():
    """
    Gracefully close the connection pool.
    Should be called on FastAPI shutdown.
    """
    global pool
    if pool:
        await pool.close()
        pool = None
        logger.info("Postgres connection pool closed.")


@asynccontextmanager
async def get_db():
    """
    Dependency for FastAPI routes.
    Provides a single pooled connection.
    Ensures pool is initialized before acquiring.
    """
    global pool
    if pool is None:
        raise RuntimeError("Connection pool is not initialized. Call init_pool() first.")

    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)


async def query_historical_trends(db):
    """
    Legacy query for monthly sales trends.
    """
    query = """
        SELECT DATE_TRUNC('month', sale_date)::DATE AS month,
               SUM(total_daily_sales) AS total_revenue
        FROM daily_sales_summary
        GROUP BY month
        ORDER BY month
    """
    try:
        results = await db.fetch(query)
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"query_historical_trends failed: {str(e)}")
        raise


async def get_dynamic_query_results(
    dimension: str,
    metric: str,
    time_period: str,
    filters: dict = None,
    db=None
):
    """
    Dynamically builds and executes a SQL query against historical_monthly_sales.
    Supports user-defined dimensions, metrics, time periods, and filters.
    """

    # Input validation
    valid_dimensions = {
        "shop_name": "shop_name",
        "item_category_name": "item_category_name",
        "month": "DATE_TRUNC('month', date)::DATE",
    }
    valid_metrics = {
        "total_revenue": "SUM(monthly_revenue)",
        "total_demand": "SUM(item_cnt_month)",
    }

    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension: {dimension}")
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}")

    # Build query
    select_clause = f"{valid_dimensions[dimension]} AS {dimension}"
    metric_clause = f"{valid_metrics[metric]} AS {metric}"
    base_query = f"SELECT {select_clause}, {metric_clause} FROM historical_monthly_sales"

    where_clauses = []
    params = []

    # Time period filter
    if time_period == "last_quarter":
        start_date = datetime.now() - timedelta(days=90)
    elif time_period == "last_month":
        start_date = datetime.now() - timedelta(days=30)
    elif time_period == "year_to_date":
        start_date = datetime(datetime.now().year, 1, 1)
    else:
        raise ValueError(f"Invalid time_period: {time_period}")

    where_clauses.append(f"date >= ${len(params)+1}")
    params.append(start_date)

    # Additional filters
    if filters:
        for key, value in filters.items():
            if key in ["shop_name", "item_category_name"] and isinstance(value, list):
                where_clauses.append(f"{key} = ANY(${len(params)+1}::text[])")
                params.append(value)
            else:
                raise ValueError(f"Invalid filter key or value: {key}={value}")

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)

    group_by_clause = f"GROUP BY {dimension}"
    order_by_clause = f"ORDER BY {metric} DESC"
    full_query = f"{base_query} {group_by_clause} {order_by_clause} LIMIT 20"

    try:
        results = await db.fetch(full_query, *params)
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Dynamic query failed: {str(e)} | Query: {full_query} | Params: {params}")
        raise RuntimeError(f"Dynamic query failed: {str(e)}")
