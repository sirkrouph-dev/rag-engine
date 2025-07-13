"""
Comprehensive error handling integration for RAG Engine.
Applies circuit breakers, retry logic, and graceful degradation throughout the system.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from datetime import datetime

from .reliability import (
    CircuitBreaker, RetryHandler, HealthChecker, GracefulDegradation,
    CircuitBreakerConfig, RetryConfig, CircuitBreakerOpenError, RetryExhaustedException
)

logger = logging.getLogger(__name__)


class ErrorHandlingIntegration:
    """Comprehensive error handling integration for the RAG engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize reliability components with configuration
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.health_checker = HealthChecker()
        self.graceful_degradation = GracefulDegradation()
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "circuit_breaker_trips": 0,
            "retry_exhaustions": 0,
            "fallback_activations": 0,
            "last_error_time": None
        }
        
        # Initialize components
        self._setup_circuit_breakers()
        self._setup_retry_handlers()
        self._setup_fallbacks()
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for different operations."""
        circuit_config = self.config.get("circuit_breaker", {})
        
        # LLM operations
        llm_config = CircuitBreakerConfig(
            failure_threshold=circuit_config.get("llm_failure_threshold", 3),
            recovery_timeout=circuit_config.get("llm_recovery_timeout", 30),
            expected_exceptions=[Exception]
        )
        self.circuit_breakers["llm"] = CircuitBreaker(llm_config)
        
        # Embedding operations
        embedding_config = CircuitBreakerConfig(
            failure_threshold=circuit_config.get("embedding_failure_threshold", 5),
            recovery_timeout=circuit_config.get("embedding_recovery_timeout", 60),
            expected_exceptions=[Exception]
        )
        self.circuit_breakers["embedding"] = CircuitBreaker(embedding_config)
        
        # Vector store operations
        vectorstore_config = CircuitBreakerConfig(
            failure_threshold=circuit_config.get("vectorstore_failure_threshold", 3),
            recovery_timeout=circuit_config.get("vectorstore_recovery_timeout", 30),
            expected_exceptions=[Exception]
        )
        self.circuit_breakers["vectorstore"] = CircuitBreaker(vectorstore_config)
        
        # External API operations
        api_config = CircuitBreakerConfig(
            failure_threshold=circuit_config.get("api_failure_threshold", 5),
            recovery_timeout=circuit_config.get("api_recovery_timeout", 60),
            expected_exceptions=[Exception]
        )
        self.circuit_breakers["external_api"] = CircuitBreaker(api_config)
    
    def _setup_retry_handlers(self):
        """Setup retry handlers for different operations."""
        retry_config = self.config.get("retry", {})
        
        # LLM operations - shorter retries due to potential cost
        llm_retry_config = RetryConfig(
            max_attempts=retry_config.get("llm_max_attempts", 2),
            backoff_factor=retry_config.get("llm_backoff_factor", 1.5),
            max_delay=retry_config.get("llm_max_delay", 30.0)
        )
        self.retry_handlers["llm"] = RetryHandler(llm_retry_config)
        
        # Embedding operations
        embedding_retry_config = RetryConfig(
            max_attempts=retry_config.get("embedding_max_attempts", 3),
            backoff_factor=retry_config.get("embedding_backoff_factor", 2.0),
            max_delay=retry_config.get("embedding_max_delay", 60.0)
        )
        self.retry_handlers["embedding"] = RetryHandler(embedding_retry_config)
        
        # Vector store operations
        vectorstore_retry_config = RetryConfig(
            max_attempts=retry_config.get("vectorstore_max_attempts", 3),
            backoff_factor=retry_config.get("vectorstore_backoff_factor", 2.0),
            max_delay=retry_config.get("vectorstore_max_delay", 30.0)
        )
        self.retry_handlers["vectorstore"] = RetryHandler(vectorstore_retry_config)
        
        # Network operations
        network_retry_config = RetryConfig(
            max_attempts=retry_config.get("network_max_attempts", 3),
            backoff_factor=retry_config.get("network_backoff_factor", 2.0),
            max_delay=retry_config.get("network_max_delay", 120.0)
        )
        self.retry_handlers["network"] = RetryHandler(network_retry_config)
    
    def _setup_fallbacks(self):
        """Setup fallback mechanisms for graceful degradation."""
        
        # LLM fallback - use simpler model or cached response
        def llm_fallback(*args, **kwargs):
            logger.warning("LLM fallback activated - using cached or simple response")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        self.graceful_degradation.register_fallback("llm_generate", llm_fallback)
        
        # Embedding fallback - use cached embeddings or simple text matching
        def embedding_fallback(*args, **kwargs):
            logger.warning("Embedding fallback activated - using simple text matching")
            return []  # Return empty list for embeddings
        
        self.graceful_degradation.register_fallback("embedding_generate", embedding_fallback)
        
        # Vector store fallback - use simple text search
        def vectorstore_fallback(*args, **kwargs):
            logger.warning("Vector store fallback activated - using simple search")
            return []  # Return empty results
        
        self.graceful_degradation.register_fallback("vectorstore_query", vectorstore_fallback)
    
    def create_protected_method(self, operation_type: str, fallback_result: Any = None):
        """Create a decorator that applies comprehensive error handling."""
        
        def protection_decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{operation_type}_{func.__name__}"
                
                try:
                    # Apply circuit breaker protection
                    circuit_breaker = self.circuit_breakers.get(operation_type)
                    if circuit_breaker:
                        # Apply retry logic within circuit breaker
                        retry_handler = self.retry_handlers.get(operation_type)
                        if retry_handler:
                            protected_func = retry_handler(func)
                        else:
                            protected_func = func
                        
                        result = await circuit_breaker.call(protected_func, *args, **kwargs)
                        return result
                    else:
                        # Just apply retry logic
                        retry_handler = self.retry_handlers.get(operation_type)
                        if retry_handler:
                            return await retry_handler(func)(*args, **kwargs)
                        else:
                            return await func(*args, **kwargs)
                
                except CircuitBreakerOpenError as e:
                    self.error_stats["circuit_breaker_trips"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Circuit breaker open for {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return await self._handle_graceful_degradation(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
                
                except RetryExhaustedException as e:
                    self.error_stats["retry_exhaustions"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Retry exhausted for {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return await self._handle_graceful_degradation(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
                
                except Exception as e:
                    self.error_stats["total_errors"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Unexpected error in {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return await self._handle_graceful_degradation(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                operation_name = f"{operation_type}_{func.__name__}"
                
                try:
                    # Apply circuit breaker protection
                    circuit_breaker = self.circuit_breakers.get(operation_type)
                    if circuit_breaker:
                        # Apply retry logic within circuit breaker
                        retry_handler = self.retry_handlers.get(operation_type)
                        if retry_handler:
                            protected_func = retry_handler(func)
                        else:
                            protected_func = func
                        
                        result = circuit_breaker.call(protected_func, *args, **kwargs)
                        return result
                    else:
                        # Just apply retry logic
                        retry_handler = self.retry_handlers.get(operation_type)
                        if retry_handler:
                            return retry_handler(func)(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                
                except CircuitBreakerOpenError as e:
                    self.error_stats["circuit_breaker_trips"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Circuit breaker open for {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return self._handle_graceful_degradation_sync(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
                
                except RetryExhaustedException as e:
                    self.error_stats["retry_exhaustions"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Retry exhausted for {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return self._handle_graceful_degradation_sync(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
                
                except Exception as e:
                    self.error_stats["total_errors"] += 1
                    self.error_stats["last_error_time"] = datetime.now()
                    logger.error(f"Unexpected error in {operation_name}: {e}")
                    
                    # Try graceful degradation
                    return self._handle_graceful_degradation_sync(
                        operation_name, fallback_result, e, *args, **kwargs
                    )
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return protection_decorator
    
    async def _handle_graceful_degradation(self, operation_name: str, fallback_result: Any, 
                                         exception: Exception, *args, **kwargs):
        """Handle graceful degradation for async operations."""
        try:
            result = await self.graceful_degradation._handle_fallback(
                operation_name, fallback_result, exception, *args, **kwargs
            )
            self.error_stats["fallback_activations"] += 1
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
            # Return safe default or re-raise
            if fallback_result is not None:
                return fallback_result
            raise exception
    
    def _handle_graceful_degradation_sync(self, operation_name: str, fallback_result: Any, 
                                        exception: Exception, *args, **kwargs):
        """Handle graceful degradation for sync operations."""
        try:
            result = self.graceful_degradation._handle_fallback_sync(
                operation_name, fallback_result, exception, *args, **kwargs
            )
            self.error_stats["fallback_activations"] += 1
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
            # Return safe default or re-raise
            if fallback_result is not None:
                return fallback_result
            raise exception
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status including error statistics."""
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }
        
        return {
            "error_handling": {
                "status": "healthy" if self.error_stats["total_errors"] < 10 else "degraded",
                "error_statistics": self.error_stats.copy(),
                "circuit_breakers": circuit_breaker_status,
                "fallbacks_registered": len(self.graceful_degradation.fallback_handlers)
            }
        }
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_stats = {
            "total_errors": 0,
            "circuit_breaker_trips": 0,
            "retry_exhaustions": 0,
            "fallback_activations": 0,
            "last_error_time": None
        }


# Convenience decorators for common operations
def llm_protected(error_handler: ErrorHandlingIntegration, fallback_result: str = None):
    """Decorator for LLM operations with comprehensive error handling."""
    return error_handler.create_protected_method("llm", fallback_result)


def embedding_protected(error_handler: ErrorHandlingIntegration, fallback_result: List = None):
    """Decorator for embedding operations with comprehensive error handling."""
    if fallback_result is None:
        fallback_result = []
    return error_handler.create_protected_method("embedding", fallback_result)


def vectorstore_protected(error_handler: ErrorHandlingIntegration, fallback_result: List = None):
    """Decorator for vector store operations with comprehensive error handling."""
    if fallback_result is None:
        fallback_result = []
    return error_handler.create_protected_method("vectorstore", fallback_result)


def external_api_protected(error_handler: ErrorHandlingIntegration, fallback_result: Any = None):
    """Decorator for external API operations with comprehensive error handling."""
    return error_handler.create_protected_method("external_api", fallback_result)


# Example usage in RAG components
class ProtectedRAGExample:
    """Example of how to apply error handling to RAG components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.error_handler = ErrorHandlingIntegration(config)
    
    @llm_protected(error_handler=None, fallback_result="I'm sorry, I cannot process your request right now.")
    async def generate_response(self, prompt: str) -> str:
        """Generate LLM response with full error protection."""
        # Your actual LLM call here
        # This will be protected by circuit breaker, retry logic, and fallbacks
        pass
    
    @embedding_protected(error_handler=None, fallback_result=[])
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with full error protection."""
        # Your actual embedding call here
        # This will be protected by circuit breaker, retry logic, and fallbacks
        pass
    
    @vectorstore_protected(error_handler=None, fallback_result=[])
    async def query_vectorstore(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Query vector store with full error protection."""
        # Your actual vector store query here
        # This will be protected by circuit breaker, retry logic, and fallbacks
        pass 