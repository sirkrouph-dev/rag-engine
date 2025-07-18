"""
Production caching integration for RAG Engine.
Handles Redis caching for rate limiting, session management, and response caching.
"""
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class CacheProvider:
    """Base class for cache providers."""
    
    async def connect(self):
        """Connect to the cache."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from the cache."""
        raise NotImplementedError
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        raise NotImplementedError
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value with optional TTL."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter."""
        raise NotImplementedError
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        raise NotImplementedError
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        raise NotImplementedError


class RedisProvider(CacheProvider):
    """Redis cache provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis = None
        
    async def connect(self):
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            
            self.redis = redis.Redis(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                password=self.config.get("password"),
                db=self.config.get("db", 0),
                decode_responses=True,
                socket_connect_timeout=self.config.get("connect_timeout", 5),
                socket_timeout=self.config.get("socket_timeout", 5),
                retry_on_timeout=True,
                max_connections=self.config.get("max_connections", 20)
            )
            
            # Test connection
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.config.get('host', 'localhost')}:{self.config.get('port', 6379)}")
            
        except ImportError:
            logger.error("redis package not installed. Run 'pip install redis'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            if self.redis is None:
                return None
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value with optional TTL."""
        try:
            if self.redis is None:
                return False
            if ttl:
                return await self.redis.setex(key, ttl, value)
            else:
                return await self.redis.set(key, value)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            if self.redis is None:
                return False
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            if self.redis is None:
                return False
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter."""
        try:
            if self.redis is None:
                return 0
            result = await self.redis.incrby(key, amount)
            if ttl and result == amount:  # First time setting the key
                await self.redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Redis INCREMENT error for key {key}: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        try:
            if self.redis is None:
                return False
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        try:
            if self.redis is None:
                return []
            return await self.redis.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS error for pattern {pattern}: {e}")
            return []


class InMemoryProvider(CacheProvider):
    """In-memory cache provider for development."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = config.get("cleanup_interval", 60)  # seconds
        self._cleanup_task = None
    
    async def connect(self):
        """Connect to in-memory cache."""
        logger.info("Using in-memory cache provider")
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def disconnect(self):
        """Disconnect from in-memory cache."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.cache.clear()
    
    async def _cleanup_expired(self):
        """Cleanup expired keys."""
        while True:
            try:
                now = time.time()
                expired_keys = [
                    key for key, data in self.cache.items()
                    if data.get("expires_at") and data["expires_at"] <= now
                ]
                for key in expired_keys:
                    del self.cache[key]
                
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        data = self.cache.get(key)
        if not data:
            return None
        
        # Check expiration
        if data.get("expires_at") and data["expires_at"] <= time.time():
            del self.cache[key]
            return None
        
        return data["value"]
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value with optional TTL."""
        data = {"value": value}
        if ttl:
            data["expires_at"] = time.time() + ttl
        
        self.cache[key] = data
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        data = self.cache.get(key)
        if not data:
            return False
        
        # Check expiration
        if data.get("expires_at") and data["expires_at"] <= time.time():
            del self.cache[key]
            return False
        
        return True
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter."""
        data = self.cache.get(key, {"value": "0"})
        
        # Check expiration
        if data.get("expires_at") and data["expires_at"] <= time.time():
            data = {"value": "0"}
        
        try:
            current_value = int(data["value"])
            new_value = current_value + amount
            
            new_data = {"value": str(new_value)}
            if ttl:
                new_data["expires_at"] = float(time.time() + ttl)
            
            self.cache[key] = new_data
            return new_value
        except (ValueError, TypeError):
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if key in self.cache:
            self.cache[key]["expires_at"] = time.time() + ttl
            return True
        return False
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]


class ProductionCacheManager:
    """Production cache manager with multiple provider support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = self._create_provider()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        # For synchronous operations
        self._loop = None
        self._cache_store = {}  # Simple in-memory store for sync operations
        self._rate_limiter = {}  # Simple rate limiter for sync operations
    
    def _create_provider(self) -> CacheProvider:
        """Create cache provider based on configuration."""
        provider_type = self.config.get("provider", "memory").lower()
        
        if provider_type == "redis":
            return RedisProvider(self.config.get("redis", {}))
        elif provider_type == "memory":
            return InMemoryProvider(self.config.get("memory", {}))
        else:
            raise ValueError(f"Unsupported cache provider: {provider_type}")
    
    def _get_or_create_loop(self):
        """Get or create event loop for async operations."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    # Synchronous wrapper methods for test compatibility
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Synchronous set operation."""
        try:
            # For simple in-memory operations, use local store
            if isinstance(value, (dict, list)):
                import json
                value = json.dumps(value)
            self._cache_store[key] = {
                "value": value,
                "expires": float(time.time() + (ttl or 300)) if ttl else None
            }
            self.stats["sets"] += 1
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache SET error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Synchronous get operation."""
        try:
            if key not in self._cache_store:
                self.stats["misses"] += 1
                return None
            
            item = self._cache_store[key]
            
            # Check expiration
            if item["expires"] and time.time() > item["expires"]:
                del self._cache_store[key]
                self.stats["misses"] += 1
                return None
            
            self.stats["hits"] += 1
            
            # Try to parse as JSON if it looks like JSON
            value = item["value"]
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    import json
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            return value
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache GET error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Synchronous delete operation."""
        try:
            if key in self._cache_store:
                del self._cache_store[key]
                self.stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache DELETE error: {e}")
            return False
    
    def check_rate_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Synchronous rate limit check."""
        try:
            key = f"rate_limit:{identifier}"
            current_time = time.time()
            
            if key not in self._rate_limiter:
                self._rate_limiter[key] = {"count": 0, "window_start": current_time}
            
            rate_data = self._rate_limiter[key]
            
            # Reset window if expired
            if current_time - rate_data["window_start"] > window:
                rate_data["count"] = 0
                rate_data["window_start"] = current_time
            
            # Check limit
            if rate_data["count"] >= limit:
                return False
            
            # Increment count
            rate_data["count"] += 1
            return True
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Fail open
    
    def cache_response(self, key: str, response_data: Dict[str, Any], ttl: int = 300) -> bool:
        """Synchronous response caching."""
        return self.set(f"response:{key}", response_data, ttl)
    
    def get_cached_response(self, key: str) -> Optional[Dict[str, Any]]:
        """Synchronous cached response retrieval."""
        return self.get(f"response:{key}")
    
    def cache_embeddings_sync(self, text_hash: str, embeddings: List[float], ttl: int = 86400) -> bool:
        """Synchronous embedding caching."""
        return self.set(f"embeddings:{text_hash}", {"embeddings": embeddings}, ttl)
    
    def get_cached_embeddings_sync(self, text_hash: str) -> Optional[List[float]]:
        """Synchronous cached embeddings retrieval."""
        data = self.get(f"embeddings:{text_hash}")
        return data.get("embeddings") if data else None
    
    def cleanup_expired(self):
        """Synchronous cleanup of expired items."""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self._cache_store.items():
            if item["expires"] and current_time > item["expires"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache_store[key]
    
    @property
    def cache_store(self):
        """Property to access cache store for testing."""
        return self._cache_store
    
    @property
    def rate_limiter(self):
        """Property to access rate limiter for testing."""
        return self._rate_limiter

    @property
    def provider_type(self) -> str:
        """Get provider type as string."""
        return self.config.get("provider", "memory")

    # Additional methods for test compatibility
    def cache_response_with_hash(self, input_data: Dict[str, Any], response_data: Dict[str, Any]) -> str:
        """Cache response with hash key."""
        cache_key = self.create_cache_key(input_data)
        self.cache_response(cache_key, response_data)
        return cache_key
    
    def cache_embedding(self, text: str, embeddings: List[float], model: str = "default") -> bool:
        """Cache embeddings for text with model."""
        text_hash = self.hash_text(f"{text}:{model}")
        return self.cache_embeddings_sync(text_hash, embeddings)
    
    def get_cached_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached embeddings for text with model."""
        text_hash = self.hash_text(f"{text}:{model}")
        return self.get_cached_embeddings_sync(text_hash)
    
    def cache_session_sync(self, session_id: str, session_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Synchronous session caching."""
        return self.set(f"session:{session_id}", session_data, ttl)
    
    def get_cached_session_sync(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous cached session retrieval."""
        return self.get(f"session:{session_id}")
    
    def invalidate_session_sync(self, session_id: str) -> bool:
        """Synchronous session invalidation."""
        return self.delete(f"session:{session_id}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.get_stats()
    
    def cleanup_expired_sync(self) -> int:
        """Synchronous cleanup of expired items."""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self._cache_store.items():
            if item["expires"] and current_time > item["expires"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache_store[key]
        
        return len(expired_keys)
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache_store)
    
    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        return [key for key in self._cache_store.keys() if fnmatch.fnmatch(key, pattern)]
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Synchronous increment operation."""
        try:
            current_value = self.get(key)
            if current_value is None:
                new_value = amount
            else:
                try:
                    new_value = int(current_value) + amount
                except (ValueError, TypeError):
                    new_value = amount
            
            self.set(key, str(new_value), ttl)
            return new_value
        except Exception as e:
            logger.error(f"Increment error: {e}")
            return 0

    def decrement(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Synchronous decrement operation."""
        return self.increment(key, -amount, ttl)
    
    def get_cached_response_by_hash(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response by hash key."""
        cache_key = self.create_cache_key(input_data)
        return self.get_cached_response(cache_key)
    
    def delete_by_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        keys = self.get_keys_by_pattern(pattern)
        deleted = 0
        for key in keys:
            if self.delete(key):
                deleted += 1
        return deleted

    async def initialize(self):
        """Initialize the cache connection."""
        await self.provider.connect()
        logger.info("Production cache initialized successfully")
    
    async def shutdown(self):
        """Shutdown the cache connection."""
        await self.provider.disconnect()
        logger.info("Production cache shutdown completed")
    
    # Async methods (original implementation)
    async def get_async(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            value = await self.provider.get(key)
            if value is not None:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return value
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache GET error: {e}")
            return None
    
    async def set_async(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            result = await self.provider.set(key, value, ttl)
            if result:
                self.stats["sets"] += 1
            return result
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache SET error: {e}")
            return False
    
    async def delete_async(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            result = await self.provider.delete(key)
            if result:
                self.stats["deletes"] += 1
            return result
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache DELETE error: {e}")
            return False
    
    # JSON operations
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from cache."""
        value = await self.get_async(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in cache key: {key}")
        return None
    
    async def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON value in cache."""
        try:
            json_value = json.dumps(value)
            return await self.set_async(key, json_value, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            return False
    
    # Rate limiting
    async def check_rate_limit_async(self, identifier: str, limit: int, window: int) -> Dict[str, Any]:
        """Check rate limit for identifier."""
        key = f"rate_limit:{identifier}"
        
        try:
            current = await self.provider.increment(key, 1, window)
            
            remaining = max(0, limit - current)
            reset_time = int(time.time()) + window
            
            return {
                "allowed": current <= limit,
                "limit": limit,
                "remaining": remaining,
                "reset_time": reset_time,
                "current": current
            }
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if cache is down
            return {
                "allowed": True,
                "limit": limit,
                "remaining": limit - 1,
                "reset_time": int(time.time()) + window,
                "current": 1
            }
    
    # Session caching
    async def cache_session(self, session_id: str, session_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache session data."""
        key = f"session:{session_id}"
        return await self.set_json(key, session_data, ttl)
    
    async def get_cached_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        key = f"session:{session_id}"
        return await self.get_json(key)
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate cached session."""
        key = f"session:{session_id}"
        return await self.delete_async(key)
    
    # Response caching
    def create_cache_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Create deterministic hash from arguments
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached_response(self, ttl: int = 300, key_prefix: str = "response"):
        """Decorator for caching function responses."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{key_prefix}:{self.create_cache_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached = await self.get_json(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set_json(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    # Embedding caching
    async def cache_embeddings(self, text_hash: str, embeddings: List[float], ttl: int = 86400) -> bool:
        """Cache embeddings for text."""
        key = f"embeddings:{text_hash}"
        return await self.set_json(key, {"embeddings": embeddings}, ttl)
    
    async def get_cached_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embeddings for text."""
        key = f"embeddings:{text_hash}"
        data = await self.get_json(key)
        return data.get("embeddings") if data else None
    
    def hash_text(self, text: str) -> str:
        """Create hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    # Cache statistics and monitoring
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
        
        return {
            "provider": self.config.get("provider", "memory"),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "errors": self.stats["errors"],
            "hit_rate": hit_rate
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    # Bulk operations
    async def get_multiple(self, keys: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple values from cache."""
        results = {}
        for key in keys:
            results[key] = await self.get_async(key)
        return results
    
    async def set_multiple(self, data: Dict[str, str], ttl: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple values in cache."""
        results = {}
        for key, value in data.items():
            results[key] = await self.set_async(key, value, ttl)
        return results
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = await self.provider.keys(pattern)
            deleted = 0
            for key in keys:
                if await self.delete_async(key):
                    deleted += 1
            return deleted
        except Exception as e:
            logger.error(f"Delete pattern error: {e}")
            return 0


# Example usage decorators
def cache_embeddings(cache_manager: ProductionCacheManager, ttl: int = 86400):
    """Decorator for caching embedding operations."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            # Create hash for text
            text_hash = cache_manager.hash_text(text)
            
            # Try to get from cache
            cached_embeddings = await cache_manager.get_cached_embeddings(text_hash)
            if cached_embeddings is not None:
                return cached_embeddings
            
            # Generate embeddings
            if asyncio.iscoroutinefunction(func):
                embeddings = await func(text, *args, **kwargs)
            else:
                embeddings = func(text, *args, **kwargs)
            
            # Cache embeddings
            await cache_manager.cache_embeddings(text_hash, embeddings, ttl)
            
            return embeddings
        return wrapper
    return decorator


def cache_llm_response(cache_manager: ProductionCacheManager, ttl: int = 3600):
    """Decorator for caching LLM responses."""
    return cache_manager.cached_response(ttl=ttl, key_prefix="llm_response") 