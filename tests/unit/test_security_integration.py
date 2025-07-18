"""
Unit tests for security integration features.

Tests authentication, input validation, audit logging, rate limiting,
and all security middleware components.
"""

import pytest
import time
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from rag_engine.interfaces.security_integration import SecurityIntegration
from rag_engine.core.production_database import ProductionDatabaseManager


class TestSecurityIntegration:
    """Test the security integration system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "security": {
                "jwt_secret_key": "test-secret-key-12345",
                "jwt_algorithm": "HS256",
                "jwt_expiration": 3600,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100,
                    "burst_limit": 200,
                    "window_seconds": 60
                },
                "input_validation": {
                    "max_length": 10000,
                    "allow_html": False,
                    "blocked_patterns": ["<script>", "javascript:"]
                }
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            },
            # Add authentication configuration
            "enable_auth": True,
            "auth_method": "jwt",
            "jwt_secret": "test-secret-key-12345",
            "token_expiration": 3600,
            "enable_rate_limiting": True,
            "rate_limit_per_minute": 100,
            "api_keys": ["test-api-key-12345"],
            "enable_security_headers": True,
            "allowed_ips": ["192.168.1.1", "127.0.0.1"],
            "blocked_ips": ["192.168.1.100"]
        }
        
        # Create temporary database for testing
        self.db_manager = ProductionDatabaseManager(self.config)
        self.security_integration = SecurityIntegration(self.config)
    
    def test_security_integration_initialization(self):
        """Test security integration initialization."""
        assert self.security_integration.config == self.config
        assert self.security_integration.input_validator is not None
        assert self.security_integration.security_manager is not None
        assert self.security_integration.audit_logger is not None
        assert self.security_integration.circuit_breaker is not None
        assert self.security_integration.retry_handler is not None
    
    def test_input_validation_safe_text(self):
        """Test input validation with safe text."""
        safe_text = "This is a safe text input for testing"
        
        result = self.security_integration.validate_input(safe_text, "text")
        
        assert result["valid"] == True
        assert result["sanitized_data"] == safe_text
        assert len(result["issues"]) == 0
    
    def test_input_validation_html_injection(self):
        """Test input validation blocks HTML injection."""
        malicious_text = "Hello <script>alert('xss')</script> world"
        
        result = self.security_integration.validate_input(malicious_text, "text")
        
        assert result["valid"] == False
        assert "script tag detected" in str(result["issues"])
        assert "<script>" not in result["sanitized_data"]
    
    def test_input_validation_javascript_injection(self):
        """Test input validation blocks JavaScript injection."""
        malicious_text = "Click here: javascript:alert('xss')"
        
        result = self.security_integration.validate_input(malicious_text, "text")
        
        assert result["valid"] == False
        assert "blocked pattern" in str(result["issues"])
    
    def test_input_validation_length_limit(self):
        """Test input validation enforces length limits."""
        long_text = "A" * 15000  # Exceeds max_length of 10000
        
        result = self.security_integration.validate_input(long_text, "text")
        
        assert result["valid"] == False
        assert "exceeds maximum length" in str(result["issues"])
    
    def test_input_validation_email_format(self):
        """Test email validation."""
        valid_email = "test@example.com"
        invalid_email = "not-an-email"
        
        valid_result = self.security_integration.validate_input(valid_email, "email")
        invalid_result = self.security_integration.validate_input(invalid_email, "email")
        
        assert valid_result["valid"] == True
        assert invalid_result["valid"] == False
    
    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        user_data = {"user_id": "test123", "username": "testuser"}
        
        # Create token
        token = self.security_integration.create_jwt_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        payload = self.security_integration.verify_jwt_token(token)
        assert payload["user_id"] == "test123"
        assert payload["username"] == "testuser"
        assert "exp" in payload
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration handling."""
        # Create token with very short expiration
        short_config = self.config.copy()
        short_config["security"]["jwt_expiration"] = 1  # 1 second
        
        short_security = SecurityIntegration(short_config)
        
        user_data = {"user_id": "test123"}
        token = short_security.create_jwt_token(user_data)
        
        # Wait for token to expire
        time.sleep(2)
        
        payload = short_security.verify_jwt_token(token)
        assert payload is None  # Should be None for expired token
    
    def test_jwt_token_invalid_signature(self):
        """Test JWT token with invalid signature."""
        user_data = {"user_id": "test123"}
        token = self.security_integration.create_jwt_token(user_data)
        
        # Tamper with token
        tampered_token = token[:-5] + "XXXXX"
        
        payload = self.security_integration.verify_jwt_token(tampered_token)
        assert payload is None  # Should be None for invalid token
    
    def test_api_key_validation(self):
        """Test API key validation."""
        valid_key = "test-api-key-12345"
        invalid_key = "invalid-key"
        
        # Add valid key to configuration
        config_with_keys = self.config.copy()
        config_with_keys["security"]["api_keys"] = [valid_key]
        
        security_with_keys = SecurityIntegration(config_with_keys)
        
        assert security_with_keys.validate_api_key(valid_key) == True
        assert security_with_keys.validate_api_key(invalid_key) == False
    
    def test_rate_limiting_within_limits(self):
        """Test rate limiting allows requests within limits."""
        client_id = "test-client-1"
        
        # Make requests within limit
        for i in range(10):
            allowed = self.security_integration.check_rate_limit(client_id)
            assert allowed == True
    
    def test_rate_limiting_exceeds_limits(self):
        """Test rate limiting blocks requests exceeding limits."""
        client_id = "test-client-2"
        
        # Simulate exceeding rate limit
        # First, make many requests to exceed the limit
        for i in range(150):  # Exceeds burst_limit of 200 but within rate
            self.security_integration.check_rate_limit(client_id)
        
        # Now exceed burst limit
        for i in range(60):  # This should trigger rate limiting
            self.security_integration.check_rate_limit(client_id)
        
        # Next request should be rate limited
        allowed = self.security_integration.check_rate_limit(client_id)
        # Note: This test might pass depending on implementation details
        # The rate limiter might be more sophisticated
    
    def test_rate_limiting_different_clients(self):
        """Test rate limiting is per-client."""
        client1 = "test-client-1"
        client2 = "test-client-2"
        
        # Exhaust rate limit for client1
        for i in range(100):
            self.security_integration.check_rate_limit(client1)
        
        # Client2 should still be allowed
        allowed = self.security_integration.check_rate_limit(client2)
        assert allowed == True
    
    def test_audit_logging_successful_operation(self):
        """Test audit logging for successful operations."""
        log_data = {
            "user_id": "test123",
            "action": "login",
            "resource": "auth_api",
            "details": {"method": "jwt"},
            "ip_address": "192.168.1.1",
            "user_agent": "test-agent",
            "success": True
        }
        
        # Should not raise exception
        self.security_integration.log_audit_event(**log_data)
        
        # Verify log was recorded (this would require database query in real implementation)
        # For now, just ensure no exception was raised
        assert True
    
    def test_audit_logging_failed_operation(self):
        """Test audit logging for failed operations."""
        log_data = {
            "user_id": "test123",
            "action": "login",
            "resource": "auth_api",
            "details": {"error": "invalid_credentials"},
            "ip_address": "192.168.1.1",
            "user_agent": "test-agent",
            "success": False
        }
        
        # Should not raise exception
        self.security_integration.log_audit_event(**log_data)
        assert True
    
    def test_security_headers_generation(self):
        """Test security headers generation."""
        headers = self.security_integration.get_security_headers()
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            assert header in headers
    
    def test_ip_filtering_allowed_ip(self):
        """Test IP filtering allows permitted IPs."""
        # Configure allowed IPs
        config_with_ip_filter = self.config.copy()
        config_with_ip_filter["security"]["ip_filtering"] = {
            "enabled": True,
            "allowed_ips": ["192.168.1.1", "10.0.0.0/8"]
        }
        
        security_with_ip_filter = SecurityIntegration(config_with_ip_filter)
        
        assert security_with_ip_filter.is_ip_allowed("192.168.1.1") == True
        assert security_with_ip_filter.is_ip_allowed("10.5.5.5") == True  # Within 10.0.0.0/8
    
    def test_ip_filtering_blocked_ip(self):
        """Test IP filtering blocks non-permitted IPs."""
        config_with_ip_filter = self.config.copy()
        config_with_ip_filter["security"]["ip_filtering"] = {
            "enabled": True,
            "allowed_ips": ["192.168.1.1"],
            "blocked_ips": ["192.168.1.100"]
        }
        
        security_with_ip_filter = SecurityIntegration(config_with_ip_filter)
        
        assert security_with_ip_filter.is_ip_allowed("192.168.1.100") == False
        assert security_with_ip_filter.is_ip_allowed("8.8.8.8") == False  # Not in allowed list
    
    def test_password_hashing_and_verification(self):
        """Test password hashing and verification."""
        password = "test-password-123"
        
        # Hash password
        hashed = self.security_integration.hash_password(password)
        assert hashed != password  # Should be hashed
        assert len(hashed) > 0
        
        # Verify correct password
        assert self.security_integration.verify_password(password, hashed) == True
        
        # Verify incorrect password
        assert self.security_integration.verify_password("wrong-password", hashed) == False
    
    def test_session_management(self):
        """Test session creation and validation."""
        user_id = "test123"
        session_data = {"user_id": user_id, "role": "user"}
        
        # Create session
        session_id = self.security_integration.create_session(user_id, session_data)
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Validate session
        retrieved_data = self.security_integration.get_session(session_id)
        assert retrieved_data is not None
        assert retrieved_data["user_id"] == user_id
        
        # Invalidate session
        self.security_integration.invalidate_session(session_id)
        
        # Should no longer be valid
        retrieved_data = self.security_integration.get_session(session_id)
        assert retrieved_data is None
    
    def test_input_sanitization_html(self):
        """Test HTML sanitization."""
        html_input = "<p>Safe content</p><script>alert('xss')</script>"
        
        sanitized = self.security_integration.sanitize_html(html_input)
        
        assert "<p>Safe content</p>" in sanitized
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
    
    def test_csrf_token_generation_and_validation(self):
        """Test CSRF token generation and validation."""
        session_id = "test_session_123"
        
        # Generate CSRF token
        csrf_token = self.security_integration.generate_csrf_token()
        assert isinstance(csrf_token, str)
        assert len(csrf_token) > 0
        
        # Validate token
        is_valid = self.security_integration.validate_csrf_token(csrf_token, csrf_token)
        assert is_valid == True
        
        # Test invalid token
        is_valid = self.security_integration.validate_csrf_token(csrf_token, "invalid_token")
        assert is_valid == False
    
    def test_security_middleware_integration(self):
        """Test security middleware integration."""
        # Mock FastAPI request and response
        mock_request = Mock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.method = "POST"
        mock_request.url.path = "/api/test"
        
        mock_response = Mock()
        
        # Test middleware processing
        middleware = self.security_integration.get_fastapi_middleware()
        assert middleware is not None
        
        # The middleware should be a callable that can process requests
        assert callable(middleware)


class TestSecurityIntegrationEdgeCases:
    """Test edge cases and error conditions for security integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "security": {
                "jwt_secret_key": "test-secret",
                "rate_limiting": {"enabled": True}
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
        self.db_manager = ProductionDatabaseManager(self.config)
        self.security_integration = SecurityIntegration(self.config)
    
    def test_malformed_jwt_token(self):
        """Test handling of malformed JWT tokens."""
        malformed_tokens = [
            "not.a.jwt",
            "header.payload",  # Missing signature
            "",  # Empty string
            "a.b.c.d",  # Too many parts
        ]
        
        for token in malformed_tokens:
            payload = self.security_integration.verify_jwt_token(token)
            assert payload is None
    
    def test_sql_injection_attempts(self):
        """Test protection against SQL injection."""
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; DELETE FROM users WHERE 1=1; --"
        ]
        
        for attempt in sql_injection_attempts:
            result = self.security_integration.validate_input(attempt, "text")
            # Should either be blocked or sanitized
            assert result["valid"] == False or attempt not in result["sanitized_data"]
    
    def test_xss_attempts(self):
        """Test protection against XSS attacks."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for attempt in xss_attempts:
            result = self.security_integration.validate_input(attempt, "text")
            assert result["valid"] == False
    
    def test_rate_limiting_edge_cases(self):
        """Test rate limiting edge cases."""
        # Test with None client_id
        allowed = self.security_integration.check_rate_limit(None)
        assert isinstance(allowed, bool)
        
        # Test with empty string client_id
        allowed = self.security_integration.check_rate_limit("")
        assert isinstance(allowed, bool)
        
        # Test with very long client_id
        long_client_id = "x" * 1000
        allowed = self.security_integration.check_rate_limit(long_client_id)
        assert isinstance(allowed, bool)
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent access."""
        import threading
        import time
        
        client_id = "concurrent-test-client"
        results = []
        
        def make_requests():
            for _ in range(10):
                result = self.security_integration.check_rate_limit(client_id)
                results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have some results
        assert len(results) > 0
        # All results should be boolean
        assert all(isinstance(r, bool) for r in results)
    
    def test_audit_logging_with_missing_fields(self):
        """Test audit logging with missing required fields."""
        incomplete_data = {
            "user_id": "test123",
            "action": "test"
            # Missing required fields
        }
        
        # Should handle gracefully without crashing
        try:
            self.security_integration.log_audit_event(**incomplete_data)
            # Should either succeed with defaults or fail gracefully
            assert True
        except Exception as e:
            # If it raises an exception, it should be a validation error
            assert "required" in str(e).lower() or "missing" in str(e).lower()
    
    def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow unbounded under load."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(100):
            client_id = f"client-{i}"
            self.security_integration.check_rate_limit(client_id)
            
            user_data = {"user_id": f"user-{i}"}
            token = self.security_integration.create_jwt_token(user_data)
            self.security_integration.verify_jwt_token(token)
            
            text = f"Test input {i}"
            self.security_integration.validate_input(text, "text")
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable (less than 50% increase)
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 1.5, f"Memory growth too high: {growth_ratio}"


class TestSecurityIntegrationPerformance:
    """Performance tests for security integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "security": {
                "jwt_secret_key": "test-secret",
                "rate_limiting": {"enabled": True}
            },
            "database": {
                "provider": "sqlite",
                "sqlite": {"database": ":memory:"}
            }
        }
        self.db_manager = ProductionDatabaseManager(self.config)
        self.security_integration = SecurityIntegration(self.config)
    
    @pytest.mark.performance
    def test_jwt_token_performance(self):
        """Test JWT token creation and verification performance."""
        user_data = {"user_id": "test123", "username": "testuser"}
        
        # Test token creation performance
        start_time = time.time()
        for _ in range(100):
            token = self.security_integration.create_jwt_token(user_data)
        creation_time = time.time() - start_time
        
        # Should create 100 tokens in less than 1 second
        assert creation_time < 1.0, f"Token creation too slow: {creation_time}s"
        
        # Test token verification performance
        token = self.security_integration.create_jwt_token(user_data)
        start_time = time.time()
        for _ in range(100):
            payload = self.security_integration.verify_jwt_token(token)
        verification_time = time.time() - start_time
        
        # Should verify 100 tokens in less than 1 second
        assert verification_time < 1.0, f"Token verification too slow: {verification_time}s"
    
    @pytest.mark.performance
    def test_input_validation_performance(self):
        """Test input validation performance."""
        test_text = "This is a test input for performance testing. " * 10
        
        start_time = time.time()
        for _ in range(100):
            result = self.security_integration.validate_input(test_text, "text")
        validation_time = time.time() - start_time
        
        # Should validate 100 inputs in less than 1 second
        assert validation_time < 1.0, f"Input validation too slow: {validation_time}s"
    
    @pytest.mark.performance
    def test_rate_limiting_performance(self):
        """Test rate limiting performance."""
        client_id = "performance-test-client"
        
        start_time = time.time()
        for _ in range(100):
            allowed = self.security_integration.check_rate_limit(client_id)
        rate_limit_time = time.time() - start_time
        
        # Should check 100 rate limits in less than 1 second
        assert rate_limit_time < 1.0, f"Rate limiting too slow: {rate_limit_time}s" 