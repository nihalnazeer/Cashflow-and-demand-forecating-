import redis.asyncio as aioredis
import json
import logging
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize Redis connection with error handling
try:
    r = aioredis.from_url(settings.redis_url)  # Now this should work with the property
    logger.info(f"Redis client initialized with URL: {settings.redis_url}")
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {e}")
    r = None

async def cache_forecast(forecast_id: str, forecast_data: dict, ttl: int = 3600):
    """
    Caches forecast data in Redis with a forecast_id key.
    """
    if r is None:
        logger.warning("Redis not available, skipping cache")
        return False
    
    try:
        serialized_data = json.dumps(forecast_data, default=str)
        await r.setex(f"forecast:{forecast_id}", ttl, serialized_data)
        logger.info(f"Cached forecast with ID: {forecast_id}")
        return True
    except Exception as e:
        logger.error(f"Error caching forecast {forecast_id}: {e}")
        return False

async def get_cached_forecast(forecast_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves cached forecast by forecast_id.
    """
    if r is None:
        logger.warning("Redis not available, no cache retrieval")
        return None
    
    try:
        val = await r.get(f"forecast:{forecast_id}")
        if val:
            logger.info(f"Retrieved cached forecast with ID: {forecast_id}")
            return json.loads(val)
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached forecast {forecast_id}: {e}")
        return None

async def cache_insight(forecast_id: str, insight_data: dict, ttl: int = 86400):
    """
    Caches insight data in Redis with a forecast_id key.
    """
    if r is None:
        logger.warning("Redis not available, skipping cache")
        return False
    
    try:
        serialized_data = json.dumps(insight_data, default=str)
        await r.setex(f"insight:{forecast_id}", ttl, serialized_data)
        logger.info(f"Cached insight with ID: {forecast_id}")
        return True
    except Exception as e:
        logger.error(f"Error caching insight {forecast_id}: {e}")
        return False

async def get_cached_insight_from_redis(forecast_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves cached insight by forecast_id.
    """
    if r is None:
        logger.warning("Redis not available, no cache retrieval")
        return None
    
    try:
        val = await r.get(f"insight:{forecast_id}")
        if val:
            logger.info(f"Retrieved cached insight with ID: {forecast_id}")
            return json.loads(val)
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached insight {forecast_id}: {e}")
        return None

async def clear_cache_by_pattern(pattern: str = "*"):
    """
    Clear cache entries matching pattern.
    """
    if r is None:
        logger.warning("Redis not available, cannot clear cache")
        return False
    
    try:
        keys = await r.keys(pattern)
        if keys:
            await r.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries matching pattern: {pattern}")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache with pattern {pattern}: {e}")
        return False

async def get_redis_health():
    """
    Check Redis connection health.
    """
    if r is None:
        return {"status": "disconnected", "error": "Redis client not initialized"}
    
    try:
        await r.ping()
        info = await r.info()
        return {
            "status": "connected",
            "redis_version": info.get("redis_version", "unknown"),
            "memory_usage": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0)
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": "error", "error": str(e)}

# Test function to verify Redis is working
async def test_redis_connection():
    """
    Test Redis connection and basic operations.
    """
    try:
        # Test basic set/get
        test_key = "test:connection"
        test_value = {"test": "data", "timestamp": "2025-01-01"}
        
        await cache_forecast("test_forecast", test_value, 60)
        retrieved = await get_cached_forecast("test_forecast")
        
        if retrieved and retrieved.get("test") == "data":
            logger.info("Redis connection test passed")
            return True
        else:
            logger.error("Redis connection test failed - data mismatch")
            return False
            
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False