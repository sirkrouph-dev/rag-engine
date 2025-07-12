"""
Production-grade error handling, circuit breakers, and reliability patterns.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from dataclasses import dataclass, field
import threading
from collections import defaultdict
import redis
import json

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exceptions: List[Type[Exception]] = field(default_factory=list)
    success_threshold: int = 3  # For half-open state


class CircuitBreaker:
    """Production-grade circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self.state == CircuitBreakerState.OPEN and
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _record_success(self):
        """Record a successful call."""
        with self.lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
    
    def _record_failure(self, exception: Exception):
        """Record a failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker opened from HALF_OPEN state")
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker moved to HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            if not self.config.expected_exceptions or \
               any(isinstance(e, exc_type) for exc_type in self.config.expected_exceptions):
                self._record_failure(e)
            raise


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    max_delay: float = 300.0
    jitter: bool = True


class RetryHandler:
    """Production-grade retry handler with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to a function."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await self._retry_async(func, *args, **kwargs)
            else:
                return self._retry_sync(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._retry_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = min(
            self.config.backoff_factor ** attempt,
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add jitter
        
        return delay
    
    async def _retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
                    break
        
        raise RetryExhaustedException(
            f"Failed after {self.config.max_attempts} attempts"
        ) from last_exception
    
    def _retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry sync function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt - 1)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
                    break
        
        raise RetryExhaustedException(
            f"Failed after {self.config.max_attempts} attempts"
        ) from last_exception


class HealthChecker:
    """Production-grade health checking system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.components: Dict[str, Callable] = {}
        self.status_cache = {}
        self.cache_ttl = 30  # seconds
    
    def register_component(self, name: str, check_func: Callable):
        """Register a component health check."""
        self.components[name] = check_func
        logger.info(f"Registered health check for component: {name}")
    
    async def check_component(self, name: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        if name not in self.components:
            return {
                "name": name,
                "status": "unknown",
                "error": "Component not registered"
            }
        
        try:
            check_func = self.components[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            return {
                "name": name,
                "status": "healthy",
                "details": result if isinstance(result, dict) else {"status": "ok"},
                "timestamp": time.time()
            }
        
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return {
                "name": name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def check_all_components(self) -> Dict[str, Any]:
        """Check health of all registered components."""
        results = {}
        overall_status = "healthy"
        
        for name in self.components:
            result = await self.check_component(name)
            results[name] = result
            
            if result["status"] != "healthy":
                overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "components": results,
            "timestamp": time.time()
        }


class GracefulDegradation:
    """Implements graceful degradation patterns."""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_handlers[operation] = fallback_func
        logger.info(f"Registered fallback for operation: {operation}")
    
    def __call__(self, operation: str, fallback_result: Any = None):
        """Decorator for graceful degradation."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Operation {operation} failed, attempting graceful degradation: {e}")
                    return await self._handle_fallback(operation, fallback_result, e, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Operation {operation} failed, attempting graceful degradation: {e}")
                    return self._handle_fallback_sync(operation, fallback_result, e, *args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _handle_fallback(self, operation: str, fallback_result: Any, 
                              exception: Exception, *args, **kwargs) -> Any:
        """Handle fallback for async operations."""
        if operation in self.fallback_handlers:
            try:
                fallback_func = self.fallback_handlers[operation]
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for {operation}: {fallback_error}")
        
        if fallback_result is not None:
            return fallback_result
        
        # Re-raise original exception if no fallback available
        raise exception
    
    def _handle_fallback_sync(self, operation: str, fallback_result: Any, 
                             exception: Exception, *args, **kwargs) -> Any:
        """Handle fallback for sync operations."""
        if operation in self.fallback_handlers:
            try:
                fallback_func = self.fallback_handlers[operation]
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for {operation}: {fallback_error}")
        
        if fallback_result is not None:
            return fallback_result
        
        # Re-raise original exception if no fallback available
        raise exception


# Custom Exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class ProductionError(Exception):
    """Base class for production errors."""
    pass


# Production Error Handling Utilities
def setup_production_error_handling(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup production error handling components."""
    
    # Circuit breaker configuration
    circuit_config = CircuitBreakerConfig(
        failure_threshold=config.get("circuit_breaker", {}).get("failure_threshold", 5),
        recovery_timeout=config.get("circuit_breaker", {}).get("recovery_timeout", 60),
        expected_exceptions=[Exception]  # Catch all exceptions by default
    )
    
    # Retry configuration
    retry_config = RetryConfig(
        max_attempts=config.get("retry_policy", {}).get("max_attempts", 3),
        backoff_factor=config.get("retry_policy", {}).get("backoff_factor", 2),
        max_delay=config.get("retry_policy", {}).get("max_delay", 300)
    )
    
    # Setup Redis for health checking if configured
    redis_client = None
    if "redis" in config:
        try:
            redis_client = redis.Redis(
                host=config["redis"]["host"],
                port=config["redis"]["port"],
                password=config["redis"].get("password"),
                db=config["redis"].get("db", 0),
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Failed to setup Redis for health checking: {e}")
    
    return {
        "circuit_breaker": CircuitBreaker(circuit_config),
        "retry_handler": RetryHandler(retry_config),
        "health_checker": HealthChecker(redis_client),
        "graceful_degradation": GracefulDegradation()
    }
