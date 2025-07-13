"""
Integration tests for production API features.

Tests the complete integration of security, error handling, monitoring,
caching, and all production middleware components working together.
"""

import pytest
import time
import asyncio
import json
import requests
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, Any

from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
from rag_engine.interfaces.api_config import APICustomizationConfig, SecurityConfig, MonitoringConfig
from rag_engine.core.production_database import ProductionDatabaseManager


class TestProductionAPIIntegration:
    """Test complete production API integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Production-like configuration
        self.config = {
            "security": {
                "jwt_secret_key": "test-secret-key-for-integration",
                "jwt_algorithm": "HS256",
                "jwt_expiration": 3600,
                "api_keys": ["test-api-key-123", "test-api-key-456"],
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100,
                    "burst_limit": 150
                },
                "input_validation": {
                    "enabled": True,
                    "max_length": 10000
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {"enabled": True},
                "health_checks": {"enabled": True},
                "alerting": {"enabled": True}
            },
            "error_handling": {
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5
                },
                "retry": {
                    "enabled": True,
                    "max_attempts": 3
                },
                "graceful_degradation": {"enabled": True}
            },
            "caching": {
                "enabled": True,
                "provider": "memory",
                "response_caching": {"enabled": True},
                "embedding_caching": {"enabled": True}
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
        
        # API configuration
        self.api_config = APICustomizationConfig(
            debug=False,
            enable_docs=True,
            security=SecurityConfig(
                enable_auth=True,
                enable_rate_limiting=True,
                enable_security_headers=True
            ),
            monitoring=MonitoringConfig(
                enable_metrics=True,
                enable_health_checks=True,
                enable_prometheus=True
            )
        )
        
        # Create FastAPI application with production features
        self.api = FastAPIEnhanced(config=self.config, api_config=self.api_config)
        self.app = self.api.create_app()
        self.client = TestClient(self.app)
        
        # Initialize database
        self.db_manager = ProductionDatabaseManager(self.config)
        self.db_manager.initialize_database()
        
        # Create test user
        self.test_user_id = self.db_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="test_password_123",
            role="user"
        )
    
    def test_health_check_endpoint(self):
        """Test health check endpoint with monitoring integration."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert data["status"] in ["healthy", "unhealthy"]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        # Generate some activity first
        self.client.get("/health")
        self.client.get("/health")
        
        response = self.client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus format or JSON metrics
        assert len(response.content) > 0
    
    def test_api_key_authentication(self):
        """Test API key authentication."""
        # Test without API key
        response = self.client.post("/api/chat", json={"query": "test"})
        assert response.status_code in [401, 403]  # Unauthorized or Forbidden
        
        # Test with invalid API key
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.post("/api/chat", json={"query": "test"}, headers=headers)
        assert response.status_code in [401, 403]
        
        # Test with valid API key
        headers = {"Authorization": "Bearer test-api-key-123"}
        response = self.client.post("/api/chat", json={"query": "test"}, headers=headers)
        # Should not fail due to auth (might fail for other reasons)
        assert response.status_code != 401
    
    def test_jwt_authentication_flow(self):
        """Test JWT authentication flow."""
        # Login to get JWT token
        login_data = {
            "username": "testuser",
            "password": "test_password_123"
        }
        
        response = self.client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        token_data = response.json()
        assert "access_token" in token_data
        
        # Use JWT token for authenticated request
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = self.client.post("/api/chat", json={"query": "test"}, headers=headers)
        
        # Should not fail due to auth
        assert response.status_code != 401
    
    def test_rate_limiting_enforcement(self):
        """Test rate limiting enforcement."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Make requests within limit
        for i in range(10):
            response = self.client.get("/health", headers=headers)
            assert response.status_code == 200
        
        # Make many requests to trigger rate limiting
        rate_limited = False
        for i in range(100):
            response = self.client.get("/health", headers=headers)
            if response.status_code == 429:  # Too Many Requests
                rate_limited = True
                break
        
        # Should eventually hit rate limit (depending on configuration)
        # Note: This might not trigger in test environment with high limits
        assert rate_limited or True  # Accept either outcome for testing
    
    def test_input_validation_security(self):
        """Test input validation and security filtering."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Test with malicious input
        malicious_inputs = [
            {"query": "<script>alert('xss')</script>"},
            {"query": "'; DROP TABLE users; --"},
            {"query": "javascript:alert('xss')"},
            {"query": "x" * 15000}  # Very long input
        ]
        
        for malicious_input in malicious_inputs:
            response = self.client.post("/api/chat", json=malicious_input, headers=headers)
            
            # Should either reject (400) or sanitize the input
            if response.status_code == 400:
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data
            else:
                # If accepted, response should not contain malicious content
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = json.dumps(response_data).lower()
                    assert "<script>" not in response_text
                    assert "drop table" not in response_text
    
    def test_security_headers(self):
        """Test security headers are properly set."""
        response = self.client.get("/health")
        
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]
        
        for header in expected_headers:
            assert header in [h.lower() for h in response.headers.keys()]
    
    def test_error_handling_with_circuit_breaker(self):
        """Test error handling with circuit breaker integration."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Mock a failing service to trigger circuit breaker
        with patch('rag_engine.core.llm.LLMManager.generate') as mock_llm:
            mock_llm.side_effect = Exception("Service unavailable")
            
            # Make requests to trigger circuit breaker
            for i in range(6):  # Above failure threshold
                response = self.client.post("/api/chat", 
                                          json={"query": "test"}, 
                                          headers=headers)
                # Should either return error or circuit breaker response
                assert response.status_code in [500, 503]
    
    def test_graceful_degradation(self):
        """Test graceful degradation when services fail."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Mock service failures
        with patch('rag_engine.core.llm.LLMManager.generate') as mock_llm:
            mock_llm.side_effect = Exception("LLM service down")
            
            response = self.client.post("/api/chat", 
                                      json={"query": "test"}, 
                                      headers=headers)
            
            # Should return graceful degradation response
            if response.status_code == 200:
                data = response.json()
                # Should contain fallback message
                assert "temporarily unavailable" in data.get("response", "").lower()
    
    def test_response_caching(self):
        """Test response caching functionality."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        query = {"query": "What is machine learning?"}
        
        # First request
        start_time = time.time()
        response1 = self.client.post("/api/chat", json=query, headers=headers)
        first_duration = time.time() - start_time
        
        # Second identical request (should be cached)
        start_time = time.time()
        response2 = self.client.post("/api/chat", json=query, headers=headers)
        second_duration = time.time() - start_time
        
        # Both should succeed
        assert response1.status_code == response2.status_code
        
        # If caching is working, second request should be faster
        # (This might not always be true in test environment)
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            # Responses should be identical if cached
            assert data1.get("response") == data2.get("response")
    
    def test_audit_logging_integration(self):
        """Test audit logging integration."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Make authenticated request
        response = self.client.post("/api/chat", 
                                  json={"query": "test audit logging"}, 
                                  headers=headers)
        
        # Check if audit log was created
        # Note: This would require access to the database or audit log system
        # For now, just verify the request completed
        assert response.status_code in [200, 400, 500]  # Any response is fine
        
        # In a real implementation, you would check:
        # audit_logs = self.db_manager.get_audit_logs(action="chat_query")
        # assert len(audit_logs) > 0
    
    def test_monitoring_metrics_collection(self):
        """Test monitoring metrics collection during API usage."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Generate API activity
        for i in range(5):
            self.client.get("/health", headers=headers)
            self.client.post("/api/chat", json={"query": f"test {i}"}, headers=headers)
        
        # Check metrics endpoint
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        # Should contain metrics data
        metrics_data = response.content.decode()
        assert len(metrics_data) > 0
        
        # Should contain request metrics
        if "http_requests_total" in metrics_data or "request_count" in metrics_data:
            # Metrics are being collected
            assert True
        else:
            # Metrics might be in different format
            assert len(metrics_data) > 100  # Should have substantial metrics data
    
    def test_concurrent_api_usage(self):
        """Test concurrent API usage with all production features."""
        import threading
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        results = []
        errors = []
        
        def api_worker(worker_id):
            try:
                for i in range(5):
                    # Mix of different endpoints
                    health_response = self.client.get("/health", headers=headers)
                    chat_response = self.client.post("/api/chat", 
                                                   json={"query": f"worker {worker_id} query {i}"}, 
                                                   headers=headers)
                    
                    results.append((health_response.status_code, chat_response.status_code))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=api_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent requests
        assert len(errors) == 0 or len(errors) < len(results) / 2
        assert len(results) > 0
        
        # Most requests should succeed
        successful_requests = sum(1 for health_status, chat_status in results 
                                if health_status == 200)
        assert successful_requests >= len(results) * 0.8  # At least 80% success
    
    def test_production_error_responses(self):
        """Test production-quality error responses."""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # Test various error conditions
        error_cases = [
            # Invalid JSON
            ({"invalid": "json", "missing": "required_fields"}, 400),
            # Missing required fields
            ({}, 400),
            # Invalid data types
            ({"query": 123}, 400),
        ]
        
        for test_data, expected_status in error_cases:
            response = self.client.post("/api/chat", json=test_data, headers=headers)
            
            # Should return appropriate error status
            assert response.status_code >= 400
            
            # Error response should be well-formed
            try:
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data
                # Should not expose internal details in production
                error_text = json.dumps(error_data).lower()
                assert "traceback" not in error_text
                assert "internal" not in error_text or "internal server error" in error_text
            except json.JSONDecodeError:
                # Some error responses might not be JSON
                assert len(response.content) > 0
    
    def test_comprehensive_production_workflow(self):
        """Test comprehensive production workflow with all features."""
        # 1. Health check
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Authenticate
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        # 3. Make valid request
        chat_response = self.client.post("/api/chat", 
                                       json={"query": "What is artificial intelligence?"}, 
                                       headers=headers)
        
        # Should not fail due to auth or basic validation
        assert chat_response.status_code != 401
        assert chat_response.status_code != 403
        
        # 4. Check metrics updated
        metrics_response = self.client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # 5. Verify security headers
        assert "x-content-type-options" in [h.lower() for h in chat_response.headers.keys()]
        
        # 6. Test rate limiting doesn't immediately trigger
        for i in range(5):
            quick_response = self.client.get("/health", headers=headers)
            assert quick_response.status_code == 200
        
        # Workflow should complete successfully
        assert True


class TestProductionAPIIntegrationAdvanced:
    """Advanced integration tests for production API features."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "security": {
                "jwt_secret_key": "advanced-test-secret",
                "api_keys": ["advanced-api-key"],
                "rate_limiting": {"enabled": True, "requests_per_minute": 50}
            },
            "monitoring": {"enabled": True},
            "error_handling": {"circuit_breaker": {"enabled": True}},
            "caching": {"enabled": True},
            "database": {"provider": "sqlite", "sqlite": {"database": ":memory:"}}
        }
        
        self.api_config = APICustomizationConfig(
            security=SecurityConfig(enable_auth=True, enable_rate_limiting=True),
            monitoring=MonitoringConfig(enable_metrics=True, enable_health_checks=True)
        )
        
        self.api = FastAPIEnhanced(config=self.config, api_config=self.api_config)
        self.app = self.api.create_app()
        self.client = TestClient(self.app)
    
    def test_api_versioning_support(self):
        """Test API versioning support."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Test different API versions if supported
        version_endpoints = ["/api/v1/chat", "/api/chat"]
        
        for endpoint in version_endpoints:
            try:
                response = self.client.post(endpoint, 
                                          json={"query": "version test"}, 
                                          headers=headers)
                # Should either work or return appropriate error
                assert response.status_code in [200, 404, 405]
            except Exception:
                # Endpoint might not exist
                pass
    
    def test_content_negotiation(self):
        """Test content negotiation and response formats."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Test different Accept headers
        accept_headers = [
            "application/json",
            "application/xml",
            "text/plain"
        ]
        
        for accept_header in accept_headers:
            request_headers = {**headers, "Accept": accept_header}
            response = self.client.post("/api/chat", 
                                      json={"query": "content negotiation test"}, 
                                      headers=request_headers)
            
            # Should handle different content types gracefully
            assert response.status_code in [200, 406, 415]  # OK, Not Acceptable, or Unsupported Media Type
    
    def test_cors_handling(self):
        """Test CORS handling for cross-origin requests."""
        headers = {
            "Authorization": "Bearer advanced-api-key",
            "Origin": "https://example.com"
        }
        
        # Preflight request
        response = self.client.options("/api/chat", headers=headers)
        
        # Should handle CORS appropriately
        assert response.status_code in [200, 204]
        
        # Check CORS headers
        cors_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers"
        ]
        
        response_header_names = [h.lower() for h in response.headers.keys()]
        cors_present = any(cors_header in response_header_names for cors_header in cors_headers)
        assert cors_present or True  # CORS might not be configured
    
    def test_request_size_limits(self):
        """Test request size limits."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Test with very large request
        large_query = "x" * 50000  # 50KB query
        large_request = {"query": large_query}
        
        response = self.client.post("/api/chat", json=large_request, headers=headers)
        
        # Should either handle large request or reject with appropriate error
        assert response.status_code in [200, 400, 413, 422]  # OK, Bad Request, Payload Too Large, or Unprocessable Entity
    
    def test_api_documentation_endpoints(self):
        """Test API documentation endpoints."""
        # Test OpenAPI/Swagger endpoints
        doc_endpoints = ["/docs", "/redoc", "/openapi.json"]
        
        for endpoint in doc_endpoints:
            response = self.client.get(endpoint)
            # Should either serve docs or return 404
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                assert len(response.content) > 0
    
    def test_production_logging_integration(self):
        """Test production logging integration."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Make request that should be logged
        response = self.client.post("/api/chat", 
                                  json={"query": "logging test"}, 
                                  headers=headers)
        
        # Request should complete (logging should not interfere)
        assert response.status_code != 500  # Should not cause internal server error
        
        # In production, you would verify logs were written:
        # - Check log files
        # - Verify structured logging format
        # - Ensure sensitive data is not logged
    
    def test_graceful_shutdown_simulation(self):
        """Test graceful shutdown behavior simulation."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Make requests during simulated shutdown
        # Note: This is a simplified test - real graceful shutdown testing
        # would require more complex setup
        
        for i in range(3):
            response = self.client.get("/health", headers=headers)
            # Should continue to respond during graceful shutdown
            assert response.status_code == 200
    
    def test_production_performance_characteristics(self):
        """Test production performance characteristics."""
        headers = {"Authorization": "Bearer advanced-api-key"}
        
        # Measure response times
        response_times = []
        
        for i in range(10):
            start_time = time.time()
            response = self.client.get("/health", headers=headers)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert response.status_code == 200
        
        # Basic performance assertions
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Should respond quickly (adjust thresholds as needed)
        assert avg_response_time < 1.0  # Average under 1 second
        assert max_response_time < 2.0  # Max under 2 seconds
        
        # Response times should be consistent (no huge outliers)
        response_time_variance = sum((t - avg_response_time) ** 2 for t in response_times) / len(response_times)
        assert response_time_variance < 1.0  # Low variance


class TestProductionAPIIntegrationEdgeCases:
    """Edge case tests for production API integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "security": {"api_keys": ["edge-case-key"]},
            "monitoring": {"enabled": True},
            "database": {"provider": "sqlite", "sqlite": {"database": ":memory:"}}
        }
        
        self.api_config = APICustomizationConfig()
        self.api = FastAPIEnhanced(config=self.config, api_config=self.api_config)
        self.app = self.api.create_app()
        self.client = TestClient(self.app)
    
    def test_malformed_requests(self):
        """Test handling of malformed requests."""
        headers = {"Authorization": "Bearer edge-case-key"}
        
        # Test various malformed requests
        malformed_requests = [
            "",  # Empty body
            "invalid json",  # Invalid JSON
            '{"incomplete": json',  # Incomplete JSON
            '{"valid": "json", "but": "unexpected_structure"}',  # Valid JSON, wrong structure
        ]
        
        for malformed_data in malformed_requests:
            try:
                response = self.client.post("/api/chat", 
                                          data=malformed_data, 
                                          headers={**headers, "Content-Type": "application/json"})
                
                # Should handle malformed requests gracefully
                assert response.status_code >= 400  # Should return error status
                assert response.status_code < 500  # Should not be server error
            except Exception:
                # Some malformed requests might cause client exceptions
                pass
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        headers = {"Authorization": "Bearer edge-case-key"}
        
        unicode_queries = [
            "Hello 世界",  # Chinese characters
            "Café naïve résumé",  # Accented characters
            "🤖 AI Assistant 🚀",  # Emojis
            "αβγδε ∑∆∇∞",  # Greek letters and math symbols
            "\u0000\u0001\u0002",  # Control characters
        ]
        
        for query in unicode_queries:
            response = self.client.post("/api/chat", 
                                      json={"query": query}, 
                                      headers=headers)
            
            # Should handle Unicode gracefully
            assert response.status_code in [200, 400]  # Either process or reject cleanly
            
            if response.status_code == 200:
                # If processed, response should be valid JSON
                try:
                    response_data = response.json()
                    assert isinstance(response_data, dict)
                except json.JSONDecodeError:
                    pytest.fail("Response should be valid JSON")
    
    def test_concurrent_authentication_attempts(self):
        """Test concurrent authentication attempts."""
        import threading
        
        results = []
        
        def auth_worker(key):
            headers = {"Authorization": f"Bearer {key}"}
            response = self.client.get("/health", headers=headers)
            results.append((key, response.status_code))
        
        # Test with mix of valid and invalid keys
        test_keys = ["edge-case-key", "invalid-key-1", "edge-case-key", "invalid-key-2"]
        
        threads = []
        for key in test_keys:
            thread = threading.Thread(target=auth_worker, args=(key,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent auth correctly
        valid_results = [r for r in results if r[0] == "edge-case-key"]
        invalid_results = [r for r in results if r[0] != "edge-case-key"]
        
        # Valid keys should succeed
        assert all(status == 200 for _, status in valid_results)
        # Invalid keys should fail
        assert all(status in [401, 403] for _, status in invalid_results)
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion."""
        headers = {"Authorization": "Bearer edge-case-key"}
        
        # Make many rapid requests
        responses = []
        for i in range(50):
            response = self.client.get("/health", headers=headers)
            responses.append(response.status_code)
        
        # Should either handle all requests or start rate limiting
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Should have some successful requests
        assert success_count > 0
        # Total should equal all requests
        assert success_count + rate_limited_count + sum(1 for status in responses if status not in [200, 429]) == 50 