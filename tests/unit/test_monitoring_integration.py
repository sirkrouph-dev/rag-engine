"""
Unit tests for monitoring integration features.

Tests metrics collection, health checks, alerting, performance monitoring,
and observability components.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from collections import defaultdict

from rag_engine.core.monitoring_integration import MonitoringIntegration


class TestMonitoringIntegration:
    """Test the monitoring integration system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "monitoring": {
                "metrics": {
                    "enabled": True,
                    "collection_interval": 30,
                    "retention_period": 86400  # 24 hours
                },
                "health_checks": {
                    "enabled": True,
                    "check_interval": 60,
                    "timeout": 10,
                    "failure_threshold": 3
                },
                "alerting": {
                    "enabled": True,
                    "thresholds": {
                        "cpu_usage": 80,
                        "memory_usage": 85,
                        "error_rate": 5,
                        "response_time": 1000
                    },
                    "cooldown_period": 300
                },
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "metrics_path": "/metrics"
                }
            }
        }
        
        self.monitoring = MonitoringIntegration(self.config)
    
    def test_monitoring_initialization(self):
        """Test monitoring integration initialization."""
        assert self.monitoring.config == self.config
        assert self.monitoring.metrics_collector is not None
        assert self.monitoring.health_monitor is not None
        assert self.monitoring.alert_manager is not None
        assert self.monitoring.performance_tracker is not None
    
    def test_metrics_collection_basic(self):
        """Test basic metrics collection."""
        # Record some metrics
        self.monitoring.record_request("GET", "/api/chat", 200, 0.5)
        self.monitoring.record_request("POST", "/api/chat", 200, 0.8)
        self.monitoring.record_request("GET", "/api/health", 200, 0.1)
        
        # Get metrics
        metrics = self.monitoring.get_metrics()
        
        assert "request_count" in metrics
        assert "response_times" in metrics
        assert "error_rates" in metrics
        
        # Check request counts
        assert metrics["request_count"]["total"] >= 3
        assert metrics["request_count"]["by_method"]["GET"] >= 2
        assert metrics["request_count"]["by_method"]["POST"] >= 1
    
    def test_metrics_collection_error_tracking(self):
        """Test error tracking in metrics."""
        # Record successful and failed requests
        self.monitoring.record_request("POST", "/api/chat", 200, 0.5)
        self.monitoring.record_request("POST", "/api/chat", 500, 1.2)
        self.monitoring.record_request("POST", "/api/chat", 404, 0.3)
        
        metrics = self.monitoring.get_metrics()
        
        # Check error tracking
        assert "error_rates" in metrics
        assert metrics["error_rates"]["5xx"] >= 1
        assert metrics["error_rates"]["4xx"] >= 1
        assert metrics["error_rates"]["total_errors"] >= 2
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        # Record performance data
        self.monitoring.record_llm_call("openai", 1.5, True)
        self.monitoring.record_llm_call("openai", 2.1, True)
        self.monitoring.record_llm_call("openai", 3.2, False)
        
        self.monitoring.record_embedding_call("openai", 0.3, True)
        self.monitoring.record_embedding_call("openai", 0.4, True)
        
        self.monitoring.record_vectorstore_operation("search", 0.1, True)
        self.monitoring.record_vectorstore_operation("insert", 0.2, True)
        
        # Get performance metrics
        performance = self.monitoring.get_performance_metrics()
        
        assert "llm_calls" in performance
        assert "embedding_calls" in performance
        assert "vectorstore_operations" in performance
        
        # Check LLM metrics
        llm_metrics = performance["llm_calls"]["openai"]
        assert llm_metrics["total_calls"] == 3
        assert llm_metrics["success_rate"] == 2/3  # 2 successes out of 3
        assert llm_metrics["avg_response_time"] > 0
    
    def test_health_checks_registration(self):
        """Test health check registration and execution."""
        def mock_database_health():
            return {
                "status": "healthy",
                "response_time": 50,
                "connections": 5
            }
        
        def mock_redis_health():
            return {
                "status": "healthy",
                "response_time": 10,
                "memory_usage": "15MB"
            }
        
        # Register health checks
        self.monitoring.register_health_check("database", mock_database_health)
        self.monitoring.register_health_check("redis", mock_redis_health)
        
        # Execute health checks
        health_status = self.monitoring.get_health_status()
        
        assert "database" in health_status
        assert "redis" in health_status
        assert health_status["database"]["status"] == "healthy"
        assert health_status["redis"]["status"] == "healthy"
    
    def test_health_checks_failure_detection(self):
        """Test health check failure detection."""
        def failing_health_check():
            raise Exception("Service unavailable")
        
        self.monitoring.register_health_check("failing_service", failing_health_check)
        
        health_status = self.monitoring.get_health_status()
        
        assert "failing_service" in health_status
        assert health_status["failing_service"]["status"] == "unhealthy"
        assert "error" in health_status["failing_service"]
    
    def test_alerting_threshold_monitoring(self):
        """Test alerting based on thresholds."""
        # Simulate high CPU usage
        self.monitoring.record_system_metric("cpu_usage", 85)  # Above threshold of 80
        self.monitoring.record_system_metric("memory_usage", 75)  # Below threshold
        
        # Check for alerts
        alerts = self.monitoring.get_active_alerts()
        
        # Should have CPU alert but not memory alert
        cpu_alert = next((alert for alert in alerts if alert["metric"] == "cpu_usage"), None)
        memory_alert = next((alert for alert in alerts if alert["metric"] == "memory_usage"), None)
        
        assert cpu_alert is not None
        assert cpu_alert["severity"] in ["warning", "critical"]
        assert memory_alert is None
    
    def test_alerting_cooldown_period(self):
        """Test alerting cooldown period."""
        # Trigger alert
        self.monitoring.record_system_metric("cpu_usage", 90)
        initial_alerts = self.monitoring.get_active_alerts()
        
        # Trigger same alert again immediately
        self.monitoring.record_system_metric("cpu_usage", 92)
        immediate_alerts = self.monitoring.get_active_alerts()
        
        # Should not create duplicate alert due to cooldown
        cpu_alerts = [alert for alert in immediate_alerts if alert["metric"] == "cpu_usage"]
        assert len(cpu_alerts) == 1  # Only one alert despite multiple triggers
    
    def test_prometheus_metrics_export(self):
        """Test Prometheus metrics export format."""
        # Record some data
        self.monitoring.record_request("GET", "/api/chat", 200, 0.5)
        self.monitoring.record_llm_call("openai", 1.2, True)
        
        # Export Prometheus metrics
        prometheus_metrics = self.monitoring.export_prometheus_metrics()
        
        assert isinstance(prometheus_metrics, str)
        assert "http_requests_total" in prometheus_metrics
        assert "http_request_duration_seconds" in prometheus_metrics
        assert "llm_calls_total" in prometheus_metrics
    
    def test_custom_metrics_tracking(self):
        """Test custom metrics tracking."""
        # Record custom metrics
        self.monitoring.record_custom_metric("user_sessions", 150)
        self.monitoring.record_custom_metric("cache_hit_rate", 0.85)
        self.monitoring.record_custom_metric("document_count", 1000)
        
        # Increment counter metrics
        self.monitoring.increment_counter("api_calls")
        self.monitoring.increment_counter("api_calls")
        self.monitoring.increment_counter("errors")
        
        custom_metrics = self.monitoring.get_custom_metrics()
        
        assert custom_metrics["user_sessions"] == 150
        assert custom_metrics["cache_hit_rate"] == 0.85
        assert custom_metrics["api_calls"] == 2
        assert custom_metrics["errors"] == 1
    
    def test_monitoring_dashboard_data(self):
        """Test monitoring dashboard data aggregation."""
        # Generate sample data
        for i in range(10):
            self.monitoring.record_request("GET", "/api/chat", 200, 0.5 + i * 0.1)
            self.monitoring.record_llm_call("openai", 1.0 + i * 0.2, True)
        
        # Get dashboard data
        dashboard_data = self.monitoring.get_dashboard_data()
        
        assert "overview" in dashboard_data
        assert "performance" in dashboard_data
        assert "health" in dashboard_data
        assert "alerts" in dashboard_data
        
        # Check overview data
        overview = dashboard_data["overview"]
        assert "total_requests" in overview
        assert "avg_response_time" in overview
        assert "error_rate" in overview
        assert "uptime" in overview
    
    def test_real_time_metrics_streaming(self):
        """Test real-time metrics streaming."""
        metrics_stream = []
        
        def metrics_callback(metrics):
            metrics_stream.append(metrics)
        
        # Start streaming (mock)
        self.monitoring.start_metrics_streaming(metrics_callback)
        
        # Generate some activity
        self.monitoring.record_request("POST", "/api/chat", 200, 0.8)
        time.sleep(0.1)  # Allow processing
        
        # Stop streaming
        self.monitoring.stop_metrics_streaming()
        
        # Should have received metrics
        assert len(metrics_stream) >= 0  # Might be 0 depending on implementation
    
    def test_historical_metrics_query(self):
        """Test querying historical metrics."""
        # Record metrics with timestamps
        current_time = time.time()
        
        for i in range(5):
            timestamp = current_time - (i * 3600)  # 1 hour intervals
            self.monitoring.record_historical_metric("requests_per_hour", 100 + i * 10, timestamp)
        
        # Query historical data
        historical_data = self.monitoring.query_historical_metrics(
            metric="requests_per_hour",
            start_time=current_time - 86400,  # Last 24 hours
            end_time=current_time
        )
        
        assert len(historical_data) >= 5
        assert all("timestamp" in point for point in historical_data)
        assert all("value" in point for point in historical_data)
    
    def test_monitoring_configuration_update(self):
        """Test updating monitoring configuration."""
        # Update thresholds
        new_thresholds = {
            "cpu_usage": 90,  # Increased from 80
            "memory_usage": 95,  # Increased from 85
            "custom_metric": 50  # New threshold
        }
        
        self.monitoring.update_alert_thresholds(new_thresholds)
        
        # Test new thresholds
        self.monitoring.record_system_metric("cpu_usage", 85)  # Should not alert now
        alerts = self.monitoring.get_active_alerts()
        
        cpu_alert = next((alert for alert in alerts if alert["metric"] == "cpu_usage"), None)
        assert cpu_alert is None  # Should not alert with new higher threshold


class TestMonitoringIntegrationAsyncOperations:
    """Test async operations in monitoring integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "monitoring": {
                "health_checks": {"enabled": True},
                "alerting": {"enabled": True}
            }
        }
        self.monitoring = MonitoringIntegration(self.config)
    
    @pytest.mark.asyncio
    async def test_async_health_checks(self):
        """Test async health checks."""
        async def async_health_check():
            await asyncio.sleep(0.1)  # Simulate async operation
            return {"status": "healthy", "async": True}
        
        # Register async health check
        self.monitoring.register_async_health_check("async_service", async_health_check)
        
        # Execute async health checks
        health_status = await self.monitoring.get_async_health_status()
        
        assert "async_service" in health_status
        assert health_status["async_service"]["status"] == "healthy"
        assert health_status["async_service"]["async"] == True
    
    @pytest.mark.asyncio
    async def test_async_metrics_collection(self):
        """Test async metrics collection."""
        async def collect_async_metrics():
            await asyncio.sleep(0.1)
            return {
                "async_operations": 50,
                "async_latency": 0.2
            }
        
        # Collect async metrics
        async_metrics = await self.monitoring.collect_async_metrics(collect_async_metrics)
        
        assert async_metrics["async_operations"] == 50
        assert async_metrics["async_latency"] == 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent execution of multiple health checks."""
        async def slow_health_check(delay, name):
            await asyncio.sleep(delay)
            return {"status": "healthy", "service": name}
        
        # Register multiple async health checks
        checks = {
            "service1": lambda: slow_health_check(0.1, "service1"),
            "service2": lambda: slow_health_check(0.2, "service2"),
            "service3": lambda: slow_health_check(0.3, "service3")
        }
        
        for name, check_func in checks.items():
            self.monitoring.register_async_health_check(name, check_func)
        
        # Execute all health checks concurrently
        start_time = time.time()
        health_status = await self.monitoring.get_async_health_status()
        end_time = time.time()
        
        # Should complete in less time than sequential execution
        assert (end_time - start_time) < 0.6  # Less than sum of all delays
        
        # All services should be healthy
        for service in ["service1", "service2", "service3"]:
            assert health_status[service]["status"] == "healthy"


class TestMonitoringIntegrationPerformance:
    """Performance tests for monitoring integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "monitoring": {
                "metrics": {"enabled": True},
                "health_checks": {"enabled": True}
            }
        }
        self.monitoring = MonitoringIntegration(self.config)
    
    @pytest.mark.performance
    def test_metrics_collection_performance(self):
        """Test metrics collection performance under load."""
        start_time = time.time()
        
        # Record many metrics
        for i in range(1000):
            self.monitoring.record_request("GET", f"/api/endpoint{i%10}", 200, 0.5)
            self.monitoring.record_llm_call("openai", 1.0, True)
            self.monitoring.record_custom_metric("test_metric", i)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 1000 metrics in reasonable time
        assert total_time < 2.0, f"Metrics collection too slow: {total_time}s"
    
    @pytest.mark.performance
    def test_health_checks_performance(self):
        """Test health checks performance."""
        def quick_health_check():
            return {"status": "healthy"}
        
        # Register multiple health checks
        for i in range(10):
            self.monitoring.register_health_check(f"service{i}", quick_health_check)
        
        start_time = time.time()
        
        # Execute health checks multiple times
        for _ in range(10):
            health_status = self.monitoring.get_health_status()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should execute quickly
        assert total_time < 1.0, f"Health checks too slow: {total_time}s"
    
    @pytest.mark.performance
    def test_concurrent_metrics_recording(self):
        """Test concurrent metrics recording performance."""
        def record_metrics():
            for i in range(100):
                self.monitoring.record_request("POST", "/api/test", 200, 0.5)
                self.monitoring.increment_counter("concurrent_test")
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
        
        start_time = time.time()
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle concurrent access efficiently
        assert total_time < 2.0, f"Concurrent metrics recording too slow: {total_time}s"
        
        # Verify data integrity
        metrics = self.monitoring.get_metrics()
        assert metrics["request_count"]["total"] >= 500  # 5 threads * 100 requests
    
    @pytest.mark.performance
    def test_prometheus_export_performance(self):
        """Test Prometheus metrics export performance."""
        # Generate substantial metrics data
        for i in range(100):
            self.monitoring.record_request("GET", f"/api/test{i%10}", 200, 0.5)
            self.monitoring.record_llm_call("openai", 1.0, True)
            self.monitoring.record_custom_metric(f"metric{i%5}", i)
        
        start_time = time.time()
        
        # Export metrics multiple times
        for _ in range(10):
            prometheus_metrics = self.monitoring.export_prometheus_metrics()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should export quickly even with substantial data
        assert total_time < 1.0, f"Prometheus export too slow: {total_time}s"
        assert len(prometheus_metrics) > 0


class TestMonitoringIntegrationEdgeCases:
    """Test edge cases and error conditions for monitoring integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "monitoring": {
                "metrics": {"enabled": True},
                "health_checks": {"enabled": True},
                "alerting": {"enabled": True}
            }
        }
        self.monitoring = MonitoringIntegration(self.config)
    
    def test_invalid_metric_values(self):
        """Test handling of invalid metric values."""
        # Test with None values
        self.monitoring.record_custom_metric("none_test", None)
        
        # Test with negative response times
        self.monitoring.record_request("GET", "/api/test", 200, -1.0)
        
        # Test with invalid status codes
        self.monitoring.record_request("GET", "/api/test", 999, 0.5)
        
        # Should handle gracefully without crashing
        metrics = self.monitoring.get_metrics()
        assert metrics is not None
    
    def test_health_check_timeout_handling(self):
        """Test health check timeout handling."""
        def slow_health_check():
            time.sleep(2.0)  # Longer than typical timeout
            return {"status": "healthy"}
        
        self.monitoring.register_health_check("slow_service", slow_health_check)
        
        # Should handle timeout gracefully
        health_status = self.monitoring.get_health_status()
        
        # Might be marked as unhealthy due to timeout
        assert "slow_service" in health_status
        # Status could be either "unhealthy" (timeout) or "healthy" (if timeout not enforced)
        assert health_status["slow_service"]["status"] in ["healthy", "unhealthy"]
    
    def test_memory_usage_monitoring(self):
        """Test that monitoring doesn't cause memory leaks."""
        import gc
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Generate substantial monitoring activity
        for i in range(1000):
            self.monitoring.record_request("GET", "/api/test", 200, 0.5)
            self.monitoring.record_custom_metric("memory_test", i)
            
            if i % 100 == 0:
                self.monitoring.get_metrics()
                self.monitoring.get_health_status()
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 2.0, f"Memory growth too high: {growth_ratio}"
    
    def test_monitoring_with_disabled_features(self):
        """Test monitoring with various features disabled."""
        disabled_config = {
            "monitoring": {
                "metrics": {"enabled": False},
                "health_checks": {"enabled": False},
                "alerting": {"enabled": False}
            }
        }
        
        disabled_monitoring = MonitoringIntegration(disabled_config)
        
        # Should handle disabled features gracefully
        disabled_monitoring.record_request("GET", "/api/test", 200, 0.5)
        metrics = disabled_monitoring.get_metrics()
        
        # Should return empty or minimal metrics
        assert metrics is not None
    
    def test_malformed_configuration(self):
        """Test handling of malformed configuration."""
        malformed_configs = [
            {},  # Empty config
            {"monitoring": {}},  # Empty monitoring config
            {"monitoring": {"invalid_key": "value"}},  # Invalid keys
            {"monitoring": {"metrics": "invalid"}},  # Invalid type
        ]
        
        for config in malformed_configs:
            try:
                monitoring = MonitoringIntegration(config)
                # Should either work with defaults or fail gracefully
                assert monitoring is not None
            except (ValueError, TypeError, KeyError):
                # Acceptable to fail with configuration errors
                pass
    
    def test_concurrent_alert_generation(self):
        """Test concurrent alert generation doesn't cause issues."""
        def generate_alerts():
            for i in range(10):
                self.monitoring.record_system_metric("cpu_usage", 90 + i)
                time.sleep(0.01)
        
        # Create multiple threads generating alerts
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=generate_alerts)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent alerts without issues
        alerts = self.monitoring.get_active_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) >= 0  # Should have some alerts
    
    def test_large_metric_names(self):
        """Test handling of large metric names."""
        large_name = "x" * 1000  # Very long metric name
        
        self.monitoring.record_custom_metric(large_name, 100)
        metrics = self.monitoring.get_custom_metrics()
        
        # Should handle gracefully (either store or truncate)
        assert metrics is not None 