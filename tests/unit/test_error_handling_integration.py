"""
Unit tests for error handling integration features.

Tests circuit breakers, retry logic, graceful degradation, health monitoring,
and all reliability patterns.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from rag_engine.core.error_handling_integration import ErrorHandlingIntegration


class TestErrorHandlingIntegration:
    """Test the error handling integration system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "error_handling": {
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout": 60,
                    "success_threshold": 3
                },
                "retry": {
                    "max_attempts": 3,
                    "backoff_factor": 2,
                    "max_delay": 300,
                    "jitter": True
                },
                "graceful_degradation": {
                    "enabled": True,
                    "fallback_responses": {
                        "llm": "I'm temporarily unable to process your request. Please try again later.",
                        "embedding": "Embedding service temporarily unavailable.",
                        "vectorstore": "Search functionality temporarily unavailable."
                    }
                },
                "health_monitoring": {
                    "check_interval": 30,
                    "failure_threshold": 3,
                    "alert_threshold": 5
                }
            }
        }
        
        self.error_handler = ErrorHandlingIntegration(self.config)
    
    def test_error_handling_initialization(self):
        """Test error handling integration initialization."""
        assert self.error_handler.config == self.config
        assert self.error_handler.circuit_breaker is not None
        assert self.error_handler.retry_handler is not None
        assert self.error_handler.graceful_degradation is not None
        assert self.error_handler.health_monitor is not None
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state (normal operation)."""
        # Mock successful function
        mock_func = Mock(return_value="success")
        
        # Wrap with circuit breaker
        protected_func = self.error_handler.protect_with_circuit_breaker(mock_func, "test_service")
        
        # Should execute normally
        result = protected_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker tracks failures."""
        failure_count = 0
        
        def failing_func():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception(f"Failure {failure_count}")
            return "success"
        
        protected_func = self.error_handler.protect_with_circuit_breaker(failing_func, "test_service")
        
        # First few calls should fail but circuit breaker stays closed
        for i in range(3):
            with pytest.raises(Exception):
                protected_func()
        
        # Circuit breaker should still be closed (threshold is 5)
        result = protected_func()
        assert result == "success"
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        def always_failing_func():
            raise Exception("Always fails")
        
        protected_func = self.error_handler.protect_with_circuit_breaker(always_failing_func, "test_service")
        
        # Trigger failures up to threshold
        for i in range(5):
            with pytest.raises(Exception):
                protected_func()
        
        # Next call should trigger circuit breaker open
        from rag_engine.core.error_handling_integration import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            protected_func()
    
    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker half-open state and recovery."""
        failure_count = 0
        
        def intermittent_func():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:
                raise Exception("Initial failures")
            return "recovered"
        
        protected_func = self.error_handler.protect_with_circuit_breaker(intermittent_func, "test_service")
        
        # Trigger circuit breaker to open
        for i in range(5):
            with pytest.raises(Exception):
                protected_func()
        
        # Should be in open state
        from rag_engine.core.error_handling_integration import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            protected_func()
        
        # Wait for recovery timeout (mock by manipulating internal state)
        self.error_handler.circuit_breaker.reset_for_testing("test_service")
        
        # Should now succeed (half-open -> closed)
        result = protected_func()
        assert result == "recovered"
    
    def test_retry_logic_success_after_retries(self):
        """Test retry logic succeeds after initial failures."""
        attempt_count = 0
        
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        # Apply retry logic
        retried_func = self.error_handler.apply_retry_logic(flaky_func, "test_operation")
        
        result = retried_func()
        assert result == "Success on attempt 3"
        assert attempt_count == 3
    
    def test_retry_logic_exhausts_attempts(self):
        """Test retry logic exhausts all attempts."""
        def always_failing_func():
            raise Exception("Always fails")
        
        retried_func = self.error_handler.apply_retry_logic(always_failing_func, "test_operation")
        
        from rag_engine.core.error_handling_integration import RetryExhaustedException
        with pytest.raises(RetryExhaustedException):
            retried_func()
    
    def test_retry_logic_backoff_timing(self):
        """Test retry logic implements proper backoff timing."""
        attempt_times = []
        
        def timing_func():
            attempt_times.append(time.time())
            raise Exception("Timing test")
        
        retried_func = self.error_handler.apply_retry_logic(timing_func, "timing_test")
        
        start_time = time.time()
        from rag_engine.core.error_handling_integration import RetryExhaustedException
        with pytest.raises(RetryExhaustedException):
            retried_func()
        
        # Should have proper delays between attempts
        assert len(attempt_times) == 3  # max_attempts
        
        # Check backoff timing (allowing some tolerance)
        if len(attempt_times) >= 2:
            delay1 = attempt_times[1] - attempt_times[0]
            assert delay1 >= 1.0  # First retry delay should be ~2^0 = 1 second
        
        if len(attempt_times) >= 3:
            delay2 = attempt_times[2] - attempt_times[1]
            assert delay2 >= 2.0  # Second retry delay should be ~2^1 = 2 seconds
    
    def test_graceful_degradation_llm_fallback(self):
        """Test graceful degradation for LLM failures."""
        def failing_llm():
            raise Exception("LLM service unavailable")
        
        # Apply graceful degradation
        degraded_func = self.error_handler.apply_graceful_degradation(failing_llm, "llm")
        
        result = degraded_func()
        assert "temporarily unable" in result.lower()
    
    def test_graceful_degradation_embedding_fallback(self):
        """Test graceful degradation for embedding failures."""
        def failing_embedding():
            raise Exception("Embedding service down")
        
        degraded_func = self.error_handler.apply_graceful_degradation(failing_embedding, "embedding")
        
        result = degraded_func()
        assert "embedding service" in result.lower()
    
    def test_graceful_degradation_vectorstore_fallback(self):
        """Test graceful degradation for vector store failures."""
        def failing_vectorstore():
            raise Exception("Vector store connection lost")
        
        degraded_func = self.error_handler.apply_graceful_degradation(failing_vectorstore, "vectorstore")
        
        result = degraded_func()
        assert "search functionality" in result.lower()
    
    def test_graceful_degradation_unknown_service(self):
        """Test graceful degradation for unknown service types."""
        def failing_unknown():
            raise Exception("Unknown service failure")
        
        degraded_func = self.error_handler.apply_graceful_degradation(failing_unknown, "unknown_service")
        
        result = degraded_func()
        assert "service temporarily unavailable" in result.lower()
    
    def test_health_monitoring_component_registration(self):
        """Test health monitoring component registration."""
        def mock_health_check():
            return {"status": "healthy", "response_time": 50}
        
        # Register component
        self.error_handler.register_health_check("test_component", mock_health_check)
        
        # Check component is registered
        health_status = self.error_handler.get_component_health("test_component")
        assert health_status["status"] == "healthy"
        assert health_status["response_time"] == 50
    
    def test_health_monitoring_failure_detection(self):
        """Test health monitoring detects component failures."""
        failure_count = 0
        
        def flaky_health_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Health check failed")
            return {"status": "healthy"}
        
        self.error_handler.register_health_check("flaky_component", flaky_health_check)
        
        # First calls should detect failures
        health_status = self.error_handler.get_component_health("flaky_component")
        assert health_status["status"] == "unhealthy"
        
        # Later call should succeed
        health_status = self.error_handler.get_component_health("flaky_component")
        assert health_status["status"] == "healthy"
    
    def test_combined_error_handling_patterns(self):
        """Test combining multiple error handling patterns."""
        attempt_count = 0
        
        def complex_failing_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("First failure")
            elif attempt_count == 2:
                raise Exception("Second failure")
            else:
                return "Success after retries"
        
        # Apply all patterns
        protected_func = self.error_handler.protect_with_circuit_breaker(complex_failing_func, "complex_service")
        retried_func = self.error_handler.apply_retry_logic(protected_func, "complex_operation")
        degraded_func = self.error_handler.apply_graceful_degradation(retried_func, "llm")
        
        result = degraded_func()
        assert result == "Success after retries"
    
    def test_error_statistics_collection(self):
        """Test error statistics collection and reporting."""
        # Generate some errors
        def error_func():
            raise Exception("Test error")
        
        protected_func = self.error_handler.protect_with_circuit_breaker(error_func, "stats_test")
        
        # Generate errors
        for i in range(3):
            try:
                protected_func()
            except:
                pass
        
        # Check statistics
        stats = self.error_handler.get_error_statistics()
        assert "stats_test" in stats
        assert stats["stats_test"]["failure_count"] >= 3
    
    def test_decorator_syntax(self):
        """Test decorator syntax for error handling."""
        @self.error_handler.with_circuit_breaker("decorated_service")
        @self.error_handler.with_retry("decorated_operation")
        @self.error_handler.with_graceful_degradation("llm")
        def decorated_func():
            return "decorated success"
        
        result = decorated_func()
        assert result == "decorated success"
    
    def test_async_error_handling(self):
        """Test error handling with async functions."""
        async def async_failing_func():
            raise Exception("Async failure")
        
        # Apply async error handling
        protected_async = self.error_handler.protect_async_with_circuit_breaker(async_failing_func, "async_service")
        
        async def run_test():
            from rag_engine.core.error_handling_integration import CircuitBreakerOpenError
            
            # Trigger failures
            for i in range(5):
                with pytest.raises(Exception):
                    await protected_async()
            
            # Should trigger circuit breaker
            with pytest.raises(CircuitBreakerOpenError):
                await protected_async()
        
        # Run async test
        asyncio.run(run_test())


class TestErrorHandlingIntegrationEdgeCases:
    """Test edge cases and error conditions for error handling integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "error_handling": {
                "circuit_breaker": {"failure_threshold": 3},
                "retry": {"max_attempts": 2},
                "graceful_degradation": {"enabled": True}
            }
        }
        self.error_handler = ErrorHandlingIntegration(self.config)
    
    def test_circuit_breaker_with_none_function(self):
        """Test circuit breaker with None function."""
        with pytest.raises((TypeError, ValueError)):
            self.error_handler.protect_with_circuit_breaker(None, "test_service")
    
    def test_circuit_breaker_with_invalid_service_name(self):
        """Test circuit breaker with invalid service names."""
        def test_func():
            return "test"
        
        # Empty service name
        protected_func = self.error_handler.protect_with_circuit_breaker(test_func, "")
        result = protected_func()
        assert result == "test"
        
        # None service name
        with pytest.raises((TypeError, ValueError)):
            self.error_handler.protect_with_circuit_breaker(test_func, None)
    
    def test_retry_logic_with_zero_attempts(self):
        """Test retry logic with zero max attempts."""
        config_zero_retry = {
            "error_handling": {
                "retry": {"max_attempts": 0}
            }
        }
        
        error_handler = ErrorHandlingIntegration(config_zero_retry)
        
        def failing_func():
            raise Exception("Should not retry")
        
        retried_func = error_handler.apply_retry_logic(failing_func, "zero_retry_test")
        
        # Should fail immediately without retries
        with pytest.raises(Exception):
            retried_func()
    
    def test_retry_logic_with_negative_backoff(self):
        """Test retry logic with negative backoff factor."""
        config_negative_backoff = {
            "error_handling": {
                "retry": {
                    "max_attempts": 3,
                    "backoff_factor": -1  # Invalid
                }
            }
        }
        
        error_handler = ErrorHandlingIntegration(config_negative_backoff)
        
        def failing_func():
            raise Exception("Negative backoff test")
        
        retried_func = error_handler.apply_retry_logic(failing_func, "negative_backoff_test")
        
        # Should handle gracefully (use default or minimum backoff)
        from rag_engine.core.error_handling_integration import RetryExhaustedException
        with pytest.raises(RetryExhaustedException):
            retried_func()
    
    def test_graceful_degradation_with_custom_fallback(self):
        """Test graceful degradation with custom fallback function."""
        def failing_func():
            raise Exception("Custom fallback test")
        
        def custom_fallback(error):
            return f"Custom fallback for: {str(error)}"
        
        degraded_func = self.error_handler.apply_graceful_degradation(
            failing_func, 
            "custom_service", 
            fallback_func=custom_fallback
        )
        
        result = degraded_func()
        assert "Custom fallback for:" in result
    
    def test_health_monitoring_with_slow_checks(self):
        """Test health monitoring with slow health checks."""
        def slow_health_check():
            time.sleep(0.1)  # Simulate slow check
            return {"status": "healthy", "response_time": 100}
        
        self.error_handler.register_health_check("slow_component", slow_health_check)
        
        start_time = time.time()
        health_status = self.error_handler.get_component_health("slow_component")
        end_time = time.time()
        
        assert health_status["status"] == "healthy"
        assert (end_time - start_time) >= 0.1  # Should take at least 0.1 seconds
    
    def test_concurrent_circuit_breaker_access(self):
        """Test circuit breaker under concurrent access."""
        failure_count = 0
        
        def concurrent_func():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 10:
                raise Exception(f"Concurrent failure {failure_count}")
            return "success"
        
        protected_func = self.error_handler.protect_with_circuit_breaker(concurrent_func, "concurrent_service")
        
        results = []
        exceptions = []
        
        def worker():
            try:
                result = protected_func()
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have some exceptions and potentially some circuit breaker errors
        assert len(exceptions) > 0
        assert len(results) >= 0  # Might be zero if circuit breaker opens quickly
    
    def test_memory_usage_under_error_load(self):
        """Test memory usage doesn't grow under continuous errors."""
        import gc
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        def memory_test_func():
            raise Exception("Memory test error")
        
        protected_func = self.error_handler.protect_with_circuit_breaker(memory_test_func, "memory_test")
        
        # Generate many errors
        for i in range(100):
            try:
                protected_func()
            except:
                pass
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 1.5, f"Memory growth too high: {growth_ratio}"
    
    def test_error_handling_with_keyboard_interrupt(self):
        """Test error handling with KeyboardInterrupt."""
        def interrupted_func():
            raise KeyboardInterrupt("User interrupted")
        
        protected_func = self.error_handler.protect_with_circuit_breaker(interrupted_func, "interrupt_test")
        
        # KeyboardInterrupt should propagate (not be caught by error handling)
        with pytest.raises(KeyboardInterrupt):
            protected_func()
    
    def test_error_handling_with_system_exit(self):
        """Test error handling with SystemExit."""
        def exit_func():
            raise SystemExit("System exit")
        
        protected_func = self.error_handler.protect_with_circuit_breaker(exit_func, "exit_test")
        
        # SystemExit should propagate
        with pytest.raises(SystemExit):
            protected_func()


class TestErrorHandlingIntegrationPerformance:
    """Performance tests for error handling integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "error_handling": {
                "circuit_breaker": {"failure_threshold": 5},
                "retry": {"max_attempts": 3}
            }
        }
        self.error_handler = ErrorHandlingIntegration(self.config)
    
    @pytest.mark.performance
    def test_circuit_breaker_performance(self):
        """Test circuit breaker performance overhead."""
        def fast_func():
            return "fast"
        
        protected_func = self.error_handler.protect_with_circuit_breaker(fast_func, "performance_test")
        
        # Measure overhead
        start_time = time.time()
        for _ in range(1000):
            result = protected_func()
        end_time = time.time()
        
        total_time = end_time - start_time
        # Should complete 1000 calls in less than 1 second
        assert total_time < 1.0, f"Circuit breaker overhead too high: {total_time}s"
    
    @pytest.mark.performance
    def test_retry_logic_performance(self):
        """Test retry logic performance for successful operations."""
        def successful_func():
            return "success"
        
        retried_func = self.error_handler.apply_retry_logic(successful_func, "performance_test")
        
        start_time = time.time()
        for _ in range(1000):
            result = retried_func()
        end_time = time.time()
        
        total_time = end_time - start_time
        # Should complete 1000 successful calls quickly
        assert total_time < 1.0, f"Retry logic overhead too high: {total_time}s"
    
    @pytest.mark.performance
    def test_health_monitoring_performance(self):
        """Test health monitoring performance."""
        def quick_health_check():
            return {"status": "healthy"}
        
        self.error_handler.register_health_check("performance_component", quick_health_check)
        
        start_time = time.time()
        for _ in range(100):
            health_status = self.error_handler.get_component_health("performance_component")
        end_time = time.time()
        
        total_time = end_time - start_time
        # Should complete 100 health checks quickly
        assert total_time < 1.0, f"Health monitoring too slow: {total_time}s"
    
    @pytest.mark.performance
    def test_combined_patterns_performance(self):
        """Test performance of combined error handling patterns."""
        def optimized_func():
            return "optimized"
        
        # Apply all patterns
        protected_func = self.error_handler.protect_with_circuit_breaker(optimized_func, "combined_test")
        retried_func = self.error_handler.apply_retry_logic(protected_func, "combined_operation")
        degraded_func = self.error_handler.apply_graceful_degradation(retried_func, "llm")
        
        start_time = time.time()
        for _ in range(100):
            result = degraded_func()
        end_time = time.time()
        
        total_time = end_time - start_time
        # Combined patterns should still be performant
        assert total_time < 2.0, f"Combined patterns too slow: {total_time}s" 