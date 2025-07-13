#!/usr/bin/env python3
"""
Comprehensive Production Test Suite for RAG Engine

This test suite validates all production features including:
- Core RAG functionality
- Security integration
- Error handling and reliability
- Monitoring and observability
- Database operations
- Caching and performance
- API integration
- End-to-end workflows

Run with: python -m pytest tests/test_comprehensive_production.py -v
"""

import pytest
import time
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_engine.core.production_database import ProductionDatabaseManager
from rag_engine.core.production_caching import ProductionCacheManager
from rag_engine.interfaces.security_integration import SecurityIntegration
from rag_engine.core.error_handling_integration import ErrorHandlingIntegration
from rag_engine.core.monitoring_integration import MonitoringIntegration


class TestProductionReadinessValidation:
    """Validate production readiness of all components."""
    
    def setup_method(self):
        """Setup production configuration for testing."""
        self.production_config = {
            "security": {
                "jwt_secret_key": "production-test-secret-key",
                "jwt_algorithm": "HS256",
                "jwt_expiration": 3600,
                "api_keys": ["prod-test-key-1", "prod-test-key-2"],
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 1000,
                    "burst_limit": 1500
                },
                "input_validation": {
                    "enabled": True,
                    "max_length": 50000,
                    "blocked_patterns": ["<script>", "DROP TABLE", "DELETE FROM"]
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {"enabled": True, "collection_interval": 30},
                "health_checks": {"enabled": True, "check_interval": 60},
                "alerting": {
                    "enabled": True,
                    "thresholds": {
                        "cpu_usage": 80,
                        "memory_usage": 85,
                        "error_rate": 5,
                        "response_time": 2000
                    }
                }
            },
            "error_handling": {
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 10,
                    "recovery_timeout": 120
                },
                "retry": {
                    "enabled": True,
                    "max_attempts": 5,
                    "backoff_factor": 2,
                    "max_delay": 300
                },
                "graceful_degradation": {"enabled": True}
            },
            "caching": {
                "enabled": True,
                "provider": "memory",
                "default_ttl": 600,
                "max_size": 10000,
                "response_caching": {"enabled": True, "ttl": 300},
                "embedding_caching": {"enabled": True, "ttl": 3600}
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
    
    def test_all_production_components_initialize(self):
        """Test all production components can be initialized."""
        # Database manager
        db_manager = ProductionDatabaseManager(self.production_config)
        assert db_manager is not None
        db_manager.initialize_database()
        
        # Cache manager
        cache_manager = ProductionCacheManager(self.production_config)
        assert cache_manager is not None
        
        # Security integration
        security_integration = SecurityIntegration(self.production_config, db_manager)
        assert security_integration is not None
        
        # Error handling integration
        error_handling = ErrorHandlingIntegration(self.production_config)
        assert error_handling is not None
        
        # Monitoring integration
        monitoring = MonitoringIntegration(self.production_config)
        assert monitoring is not None
    
    def test_production_security_features(self):
        """Test all production security features work together."""
        db_manager = ProductionDatabaseManager(self.production_config)
        db_manager.initialize_database()
        
        security = SecurityIntegration(self.production_config, db_manager)
        
        # Test user management
        user_id = security.create_user("prod_user", "prod@example.com", "secure_password_123", "user")
        assert user_id is not None
        
        # Test authentication
        auth_result = security.authenticate_user("prod_user", "secure_password_123")
        assert auth_result is not None
        assert auth_result["user_id"] == user_id
        
        # Test JWT tokens
        token = security.create_jwt_token({"user_id": user_id, "username": "prod_user"})
        assert token is not None
        
        payload = security.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
        
        # Test input validation
        safe_input = "This is a safe input for production testing"
        result = security.validate_input(safe_input, "text")
        assert result["valid"] == True
        
        malicious_input = "<script>alert('xss')</script>"
        result = security.validate_input(malicious_input, "text")
        assert result["valid"] == False
        
        # Test rate limiting
        for i in range(10):
            allowed = security.check_rate_limit("test_client", limit=20, window=60)
            assert allowed == True
    
    def test_production_error_handling_features(self):
        """Test all production error handling features work together."""
        error_handler = ErrorHandlingIntegration(self.production_config)
        
        # Test circuit breaker
        def failing_function():
            raise Exception("Test failure")
        
        protected_func = error_handler.protect_with_circuit_breaker(failing_function, "test_service")
        
        # Should fail initially
        with pytest.raises(Exception):
            protected_func()
        
        # Test retry logic
        attempt_count = 0
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Flaky failure")
            return "success"
        
        retried_func = error_handler.apply_retry_logic(flaky_function, "flaky_operation")
        result = retried_func()
        assert result == "success"
        assert attempt_count == 3
        
        # Test graceful degradation
        def degraded_function():
            raise Exception("Service unavailable")
        
        degraded_func = error_handler.apply_graceful_degradation(degraded_function, "llm")
        result = degraded_func()
        assert "temporarily unable" in result.lower()
        
        # Test health monitoring
        def health_check():
            return {"status": "healthy", "response_time": 50}
        
        error_handler.register_health_check("test_component", health_check)
        health_status = error_handler.get_component_health("test_component")
        assert health_status["status"] == "healthy"
    
    def test_production_monitoring_features(self):
        """Test all production monitoring features work together."""
        monitoring = MonitoringIntegration(self.production_config)
        
        # Test metrics collection
        monitoring.record_request("GET", "/api/test", 200, 0.5)
        monitoring.record_request("POST", "/api/chat", 200, 1.2)
        monitoring.record_request("GET", "/api/health", 200, 0.1)
        
        metrics = monitoring.get_metrics()
        assert "request_count" in metrics
        assert metrics["request_count"]["total"] >= 3
        
        # Test performance tracking
        monitoring.record_llm_call("openai", 1.5, True)
        monitoring.record_embedding_call("openai", 0.3, True)
        monitoring.record_vectorstore_operation("search", 0.1, True)
        
        performance = monitoring.get_performance_metrics()
        assert "llm_calls" in performance
        assert "embedding_calls" in performance
        assert "vectorstore_operations" in performance
        
        # Test health checks
        def mock_health_check():
            return {"status": "healthy", "uptime": 3600}
        
        monitoring.register_health_check("production_service", mock_health_check)
        health_status = monitoring.get_health_status()
        assert "production_service" in health_status
        
        # Test alerting
        monitoring.record_system_metric("cpu_usage", 85)  # Above threshold
        alerts = monitoring.get_active_alerts()
        assert len(alerts) >= 1
        
        # Test dashboard data
        dashboard_data = monitoring.get_dashboard_data()
        assert "overview" in dashboard_data
        assert "performance" in dashboard_data
        assert "health" in dashboard_data
    
    def test_production_caching_features(self):
        """Test all production caching features work together."""
        cache_manager = ProductionCacheManager(self.production_config)
        
        # Test basic caching
        cache_manager.set("prod_test_key", "prod_test_value", ttl=300)
        value = cache_manager.get("prod_test_key")
        assert value == "prod_test_value"
        
        # Test complex data caching
        complex_data = {
            "user_id": "user123",
            "preferences": {"theme": "dark", "language": "en"},
            "history": [1, 2, 3, 4, 5]
        }
        cache_manager.set("complex_data", complex_data)
        retrieved_data = cache_manager.get("complex_data")
        assert retrieved_data == complex_data
        
        # Test response caching
        response_data = {"result": "cached response", "timestamp": time.time()}
        cache_manager.cache_response("response_key", response_data)
        cached_response = cache_manager.get_cached_response("response_key")
        assert cached_response == response_data
        
        # Test embedding caching
        embeddings = [0.1, 0.2, 0.3] * 100  # Mock embeddings
        cache_manager.cache_embedding("test text", embeddings, "text-embedding-ada-002")
        cached_embeddings = cache_manager.get_cached_embedding("test text", "text-embedding-ada-002")
        assert cached_embeddings == embeddings
        
        # Test rate limiting
        client_id = "prod_test_client"
        for i in range(10):
            allowed = cache_manager.check_rate_limit(client_id, limit=20, window=60)
            assert allowed == True
        
        # Test statistics
        stats = cache_manager.get_cache_statistics()
        assert "hits" in stats
        assert "misses" in stats
        assert "sets" in stats
    
    def test_production_database_features(self):
        """Test all production database features work together."""
        db_manager = ProductionDatabaseManager(self.production_config)
        db_manager.initialize_database()
        
        # Test user management
        user_id = db_manager.create_user(
            username="prod_db_user",
            email="proddb@example.com",
            password="secure_db_password_123",
            role="admin"
        )
        assert user_id is not None
        
        # Test user retrieval
        user = db_manager.get_user(user_id)
        assert user is not None
        assert user["username"] == "prod_db_user"
        assert user["role"] == "admin"
        
        # Test authentication
        auth_result = db_manager.authenticate_user("prod_db_user", "secure_db_password_123")
        assert auth_result is not None
        assert auth_result["user_id"] == user_id
        
        # Test session management
        session_id = db_manager.create_session(user_id, user_agent="prod_test", ip_address="127.0.0.1")
        assert session_id is not None
        
        session = db_manager.get_session(session_id)
        assert session is not None
        assert session["user_id"] == user_id
        assert session["is_active"] == True
        
        # Test audit logging
        audit_id = db_manager.log_audit_event(
            user_id=user_id,
            action="production_test",
            resource="test_api",
            details={"test": "production validation"},
            ip_address="127.0.0.1",
            user_agent="prod_test_agent"
        )
        assert audit_id is not None
        
        # Test audit retrieval
        audit_logs = db_manager.get_audit_logs(user_id=user_id)
        assert len(audit_logs) >= 1
        assert audit_logs[0]["action"] == "production_test"
        
        # Test user updates
        success = db_manager.update_user(user_id, email="updated_proddb@example.com")
        assert success == True
        
        updated_user = db_manager.get_user(user_id)
        assert updated_user["email"] == "updated_proddb@example.com"
    
    def test_production_integration_comprehensive(self):
        """Test comprehensive integration of all production features."""
        # Initialize all components
        db_manager = ProductionDatabaseManager(self.production_config)
        db_manager.initialize_database()
        
        cache_manager = ProductionCacheManager(self.production_config)
        security = SecurityIntegration(self.production_config, db_manager)
        error_handler = ErrorHandlingIntegration(self.production_config)
        monitoring = MonitoringIntegration(self.production_config)
        
        # Simulate production workflow
        # 1. User creation and authentication
        user_id = security.create_user("integration_user", "integration@example.com", "integration_password_123", "user")
        token = security.create_jwt_token({"user_id": user_id, "username": "integration_user"})
        
        # 2. Cache user session
        session_data = {"user_id": user_id, "token": token, "login_time": time.time()}
        cache_manager.cache_session(f"session_{user_id}", session_data)
        
        # 3. Simulate API requests with monitoring
        for i in range(5):
            start_time = time.time()
            
            # Simulate request processing
            time.sleep(0.01)  # Simulate processing time
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Record metrics
            monitoring.record_request("POST", "/api/chat", 200, response_time)
            
            # Cache response
            response_data = {"response": f"Integration test response {i}", "user_id": user_id}
            cache_manager.cache_response(f"response_{i}", response_data)
        
        # 4. Test error handling integration
        def integrated_failing_function():
            raise Exception("Integration test failure")
        
        protected_func = error_handler.protect_with_circuit_breaker(integrated_failing_function, "integration_service")
        degraded_func = error_handler.apply_graceful_degradation(protected_func, "llm")
        
        result = degraded_func()
        assert "temporarily unable" in result.lower()
        
        # 5. Verify all components are working together
        # Check metrics
        metrics = monitoring.get_metrics()
        assert metrics["request_count"]["total"] >= 5
        
        # Check cache
        cached_session = cache_manager.get_cached_session(f"session_{user_id}")
        assert cached_session is not None
        assert cached_session["user_id"] == user_id
        
        # Check database
        user = db_manager.get_user(user_id)
        assert user is not None
        
        # Check security
        payload = security.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
    
    def test_production_performance_benchmarks(self):
        """Test production performance benchmarks."""
        # Initialize components
        db_manager = ProductionDatabaseManager(self.production_config)
        db_manager.initialize_database()
        
        cache_manager = ProductionCacheManager(self.production_config)
        security = SecurityIntegration(self.production_config, db_manager)
        monitoring = MonitoringIntegration(self.production_config)
        
        # Performance benchmarks
        benchmarks = {}
        
        # 1. User authentication performance
        user_id = security.create_user("perf_user", "perf@example.com", "perf_password_123", "user")
        
        start_time = time.time()
        for i in range(100):
            auth_result = security.authenticate_user("perf_user", "perf_password_123")
            assert auth_result is not None
        auth_time = time.time() - start_time
        benchmarks["auth_per_second"] = 100 / auth_time
        
        # 2. Cache performance
        start_time = time.time()
        for i in range(1000):
            cache_manager.set(f"perf_key_{i}", f"perf_value_{i}")
            value = cache_manager.get(f"perf_key_{i}")
            assert value == f"perf_value_{i}"
        cache_time = time.time() - start_time
        benchmarks["cache_ops_per_second"] = 2000 / cache_time  # 2 ops per iteration
        
        # 3. Monitoring performance
        start_time = time.time()
        for i in range(500):
            monitoring.record_request("GET", "/api/test", 200, 0.1)
        monitoring_time = time.time() - start_time
        benchmarks["monitoring_records_per_second"] = 500 / monitoring_time
        
        # 4. JWT token performance
        start_time = time.time()
        tokens = []
        for i in range(100):
            token = security.create_jwt_token({"user_id": user_id, "index": i})
            tokens.append(token)
        token_creation_time = time.time() - start_time
        
        start_time = time.time()
        for token in tokens:
            payload = security.verify_jwt_token(token)
            assert payload is not None
        token_verification_time = time.time() - start_time
        
        benchmarks["jwt_creation_per_second"] = 100 / token_creation_time
        benchmarks["jwt_verification_per_second"] = 100 / token_verification_time
        
        # Performance assertions (adjust thresholds as needed)
        assert benchmarks["auth_per_second"] > 50, f"Auth too slow: {benchmarks['auth_per_second']}/s"
        assert benchmarks["cache_ops_per_second"] > 1000, f"Cache too slow: {benchmarks['cache_ops_per_second']}/s"
        assert benchmarks["monitoring_records_per_second"] > 500, f"Monitoring too slow: {benchmarks['monitoring_records_per_second']}/s"
        assert benchmarks["jwt_creation_per_second"] > 100, f"JWT creation too slow: {benchmarks['jwt_creation_per_second']}/s"
        assert benchmarks["jwt_verification_per_second"] > 200, f"JWT verification too slow: {benchmarks['jwt_verification_per_second']}/s"
        
        print("\n=== Production Performance Benchmarks ===")
        for metric, value in benchmarks.items():
            print(f"{metric}: {value:.2f}")
    
    def test_production_scalability_simulation(self):
        """Test production scalability simulation."""
        import threading
        
        # Initialize components
        db_manager = ProductionDatabaseManager(self.production_config)
        db_manager.initialize_database()
        
        cache_manager = ProductionCacheManager(self.production_config)
        security = SecurityIntegration(self.production_config, db_manager)
        monitoring = MonitoringIntegration(self.production_config)
        
        # Create test users
        user_ids = []
        for i in range(10):
            user_id = security.create_user(f"scale_user_{i}", f"scale{i}@example.com", f"password_{i}", "user")
            user_ids.append(user_id)
        
        # Concurrent operations
        results = []
        errors = []
        
        def concurrent_worker(worker_id):
            try:
                worker_results = []
                
                for i in range(20):
                    # Authenticate user
                    user_id = user_ids[worker_id % len(user_ids)]
                    username = f"scale_user_{user_id.split('_')[-1] if '_' in str(user_id) else worker_id % len(user_ids)}"
                    password = f"password_{worker_id % len(user_ids)}"
                    
                    auth_start = time.time()
                    auth_result = security.authenticate_user(username, password)
                    auth_time = time.time() - auth_start
                    
                    if auth_result:
                        # Cache operations
                        cache_key = f"worker_{worker_id}_item_{i}"
                        cache_value = f"data_{worker_id}_{i}"
                        
                        cache_start = time.time()
                        cache_manager.set(cache_key, cache_value)
                        retrieved_value = cache_manager.get(cache_key)
                        cache_time = time.time() - cache_start
                        
                        # Monitoring
                        monitoring.record_request("POST", f"/api/worker_{worker_id}", 200, auth_time + cache_time)
                        
                        worker_results.append({
                            "auth_time": auth_time,
                            "cache_time": cache_time,
                            "cache_success": retrieved_value == cache_value
                        })
                
                results.extend(worker_results)
                
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent workers
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=concurrent_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Analyze results
        assert len(errors) < len(results) * 0.1, f"Too many errors: {len(errors)}/{len(results)}"
        
        if results:
            avg_auth_time = sum(r["auth_time"] for r in results) / len(results)
            avg_cache_time = sum(r["cache_time"] for r in results) / len(results)
            cache_success_rate = sum(1 for r in results if r["cache_success"]) / len(results)
            
            # Scalability assertions
            assert avg_auth_time < 1.0, f"Auth too slow under load: {avg_auth_time}s"
            assert avg_cache_time < 0.1, f"Cache too slow under load: {avg_cache_time}s"
            assert cache_success_rate > 0.95, f"Cache success rate too low: {cache_success_rate}"
            
            print(f"\n=== Scalability Results ===")
            print(f"Operations completed: {len(results)}")
            print(f"Error rate: {len(errors)}/{len(results)} ({len(errors)/len(results)*100:.1f}%)")
            print(f"Average auth time: {avg_auth_time:.3f}s")
            print(f"Average cache time: {avg_cache_time:.3f}s")
            print(f"Cache success rate: {cache_success_rate:.1%}")


def run_comprehensive_production_tests():
    """Run all comprehensive production tests."""
    print("=" * 80)
    print("COMPREHENSIVE PRODUCTION TEST SUITE")
    print("=" * 80)
    
    # Run pytest with comprehensive reporting
    pytest_args = [
        "tests/test_comprehensive_production.py",
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("🎉 ALL PRODUCTION TESTS PASSED!")
        print("✅ RAG Engine is PRODUCTION-READY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PRODUCTION TESTS FAILED")
        print("🔧 Review failures before production deployment")
        print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_comprehensive_production_tests()
    sys.exit(exit_code) 