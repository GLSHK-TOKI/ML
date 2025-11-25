import logging
from threading import Lock
from typing import Any

from cachetools import TTLCache

from .errors import SDKError

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TTL = 600 # 10 minutes
DEFAULT_MAXSIZE = 2000 # 2000 entries

cache: TTLCache[str, dict[str, Any]] | None = None
_lock = Lock() # Thread-safe singleton lock

def configure_cache(ttl: int = DEFAULT_TTL, maxsize: int = DEFAULT_MAXSIZE):
    """Configure the shared SDK cache.

    Args:
        ttl: Cache TTL in seconds.
        maxsize: Maximum number of cache entries.

    Stores:
        - Keys: str (e.g., 'access:{collection_id}:{user_id}')
        - Values: dict (e.g., {'has_access': bool, 'content': dict} for access checks)
    """
    with _lock:
        global cache  # noqa: PLW0603 # module-level singleton
        cache = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(f"Shared cache configured with TTL={ttl}s, maxsize={maxsize}")

def invalidate_cache(key: str | None = None):
    """Invalidate specific cache key or clear entire cache.

    Args:
        key: Specific cache key to remove (optional).
    """
    if cache is None:
        return
    if key:
        cache.pop(key, None)
    else:
        cache.clear()

def get_cache() -> TTLCache[str, dict[str, Any]]:
    """Get the shared cache instance, initializing with defaults if needed.

    Returns:
        TTLCache: The configured cache instance.
    """
    if cache is None:
        configure_cache()

    if not isinstance(cache, TTLCache):
        msg = "Could not initialize cache instance"
        raise SDKError(msg)

    return cache