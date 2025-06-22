"""
Test the enhanced API frameworks with customization features.
"""
import pytest
import asyncio
import time
import json
from typing import Dict, Any

def test_enhanced_fastapi_creation():
    """Test enhanced FastAPI server creation."""
    try:
        from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
        from rag_engine.interfaces.enhanced_base_api import APICustomization, AuthMethod
        
        # Create API configuration
        api_config = APICustomization(
            host="127.0.0.1",
            port=8000,
            debug=True,
            enable_docs=True,
            enable_metrics=True,
            enable_health_checks=True,
            enable_rate_limiting=True,
            enable_compression=True,
            auth_method=AuthMethod.NONE,
            custom_headers={"X-Test": "Enhanced"}
        )
        
        # Create server
        server = FastAPIEnhanced(api_config=api_config)
        app = server.create_app()
        
        assert app is not None
        assert app.title == "Enhanced RAG Engine API"
        print("✅ Enhanced FastAPI server created successfully")
        
    except ImportError:
        pytest.skip("FastAPI or dependencies not available")


def test_enhanced_flask_creation():
    """Test enhanced Flask server creation."""
    try:
        from rag_engine.interfaces.flask_enhanced import FlaskEnhanced
        from rag_engine.interfaces.enhanced_base_api import APICustomization, AuthMethod
        
        # Create API configuration
        api_config = APICustomization(
            host="127.0.0.1",
            port=5000,
            debug=True,
            enable_metrics=True,
            enable_health_checks=True,
            enable_rate_limiting=True,
            auth_method=AuthMethod.NONE,
            custom_headers={"X-Test": "Enhanced"}
        )
        
        # Create server
        server = FlaskEnhanced(api_config=api_config)
        app = server.create_app()
        
        assert app is not None
        assert app.name == "rag_engine.interfaces.flask_enhanced"
        print("✅ Enhanced Flask server created successfully")
        
    except ImportError:
        pytest.skip("Flask or dependencies not available")


def test_enhanced_django_creation():
    """Test enhanced Django server creation."""
    try:
        from rag_engine.interfaces.django_enhanced import DjangoEnhanced
        from rag_engine.interfaces.enhanced_base_api import APICustomization, AuthMethod
        
        # Create API configuration
        api_config = APICustomization(
            host="127.0.0.1",
            port=8000,
            debug=True,
            enable_metrics=True,
            enable_health_checks=True,
            enable_rate_limiting=True,
            auth_method=AuthMethod.NONE
        )
        
        # Create server
        server = DjangoEnhanced(api_config=api_config)
        app = server.create_app()
        
        assert app is not None
        print("✅ Enhanced Django server created successfully")
        
    except ImportError:
        pytest.skip("Django or dependencies not available")


def test_enhanced_factory():
    """Test the enhanced framework factory."""
    try:
        from rag_engine.interfaces.enhanced_base_api import enhanced_factory, APICustomization, AuthMethod
        from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
        
        # Register a framework
        enhanced_factory.register_framework("test_fastapi", FastAPIEnhanced)
        
        # Check framework is registered
        frameworks = enhanced_factory.list_frameworks()
        assert "test_fastapi" in frameworks
        
        # Create server using factory
        api_config = APICustomization(
            debug=True,
            enable_metrics=True,
            auth_method=AuthMethod.NONE
        )
        
        server = enhanced_factory.create_server("test_fastapi", api_config=api_config)
        assert isinstance(server, FastAPIEnhanced)
        
        print("✅ Enhanced factory working correctly")
        
    except ImportError:
        pytest.skip("FastAPI or dependencies not available")


def test_api_customization_config():
    """Test API customization configuration."""
    from rag_engine.interfaces.enhanced_base_api import APICustomization, AuthMethod, RateLimitType
    
    # Test default configuration
    config = APICustomization()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.auth_method == AuthMethod.NONE
    assert config.enable_docs == True
    
    # Test custom configuration
    custom_config = APICustomization(
        host="localhost",
        port=9000,
        debug=True,
        auth_method=AuthMethod.API_KEY,
        api_keys=["test-key-1", "test-key-2"],
        enable_rate_limiting=True,
        rate_limit_type=RateLimitType.PER_IP,
        requests_per_minute=50,
        custom_headers={"X-Custom": "Value"}
    )
    
    assert custom_config.host == "localhost"
    assert custom_config.port == 9000
    assert custom_config.debug == True
    assert custom_config.auth_method == AuthMethod.API_KEY
    assert len(custom_config.api_keys) == 2
    assert custom_config.enable_rate_limiting == True
    assert custom_config.rate_limit_type == RateLimitType.PER_IP
    assert custom_config.requests_per_minute == 50
    assert custom_config.custom_headers["X-Custom"] == "Value"
    
    print("✅ API customization configuration working correctly")


def test_security_config():
    """Test security configuration."""
    try:
        from rag_engine.interfaces.security import SecurityConfig
        
        # Test default configuration
        config = SecurityConfig()
        assert config.enable_auth == False
        assert config.auth_method == "api_key"
        assert config.enable_rate_limiting == True
        assert config.rate_limit_per_minute == 60
        
        # Test custom configuration
        custom_config = SecurityConfig(
            enable_auth=True,
            auth_method="jwt",
            jwt_secret="test-secret",
            enable_rate_limiting=True,
            rate_limit_per_minute=100,
            cors_origins=["http://localhost:3000"],
            enable_security_headers=True
        )
        
        assert custom_config.enable_auth == True
        assert custom_config.auth_method == "jwt"
        assert custom_config.jwt_secret == "test-secret"
        assert custom_config.rate_limit_per_minute == 100
        assert "http://localhost:3000" in custom_config.cors_origins
        
        print("✅ Security configuration working correctly")
        
    except ImportError:
        pytest.skip("Security dependencies not available")


def test_monitoring_components():
    """Test monitoring components."""
    try:
        from rag_engine.interfaces.monitoring import MetricsCollector, HealthChecker
        
        # Test metrics collector
        metrics = MetricsCollector()
        metrics.record_request("GET", "/test", 200, 0.1)
        
        prometheus_metrics = metrics.generate_prometheus_metrics()
        assert "requests_total" in prometheus_metrics
        assert "request_duration" in prometheus_metrics
        
        # Test health checker
        health_checker = HealthChecker()
        health_status = health_checker.check_health()
        
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "checks" in health_status
        
        print("✅ Monitoring components working correctly")
        
    except ImportError:
        pytest.skip("Monitoring dependencies not available")


def test_enhanced_configuration_loading():
    """Test loading enhanced configuration from file."""
    import os
    from pathlib import Path
    
    # Test configuration file path
    config_path = Path(__file__).parent.parent / "configs" / "enhanced_production.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Verify configuration structure
        assert "api" in config_data
        assert "security" in config_data
        assert "middleware" in config_data
        assert "monitoring" in config_data
        assert "rag" in config_data
        
        # Verify API configuration
        api_config = config_data["api"]
        assert "framework" in api_config
        assert "host" in api_config
        assert "port" in api_config
        
        # Verify security configuration
        security_config = config_data["security"]
        assert "auth_method" in security_config
        assert "enable_rate_limiting" in security_config
        
        print("✅ Enhanced configuration file structure is valid")
    else:
        print("⚠️  Enhanced configuration file not found")


def test_cli_integration():
    """Test CLI integration with enhanced frameworks."""
    try:
        from rag_engine.interfaces.enhanced_base_api import enhanced_factory
        from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
        from rag_engine.interfaces.flask_enhanced import FlaskEnhanced
        
        # Register frameworks (simulate CLI registration)
        enhanced_factory.register_framework("fastapi", FastAPIEnhanced)
        enhanced_factory.register_framework("flask", FlaskEnhanced)
        
        # Test framework listing
        frameworks = enhanced_factory.list_frameworks()
        assert len(frameworks) >= 2
        assert "fastapi" in frameworks
        assert "flask" in frameworks
        
        print("✅ CLI integration with enhanced frameworks working")
        
    except ImportError:
        pytest.skip("Framework dependencies not available")


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Testing Enhanced API Frameworks...")
    
    test_functions = [
        test_enhanced_fastapi_creation,
        test_enhanced_flask_creation,
        test_enhanced_django_creation,
        test_enhanced_factory,
        test_api_customization_config,
        test_security_config,
        test_monitoring_components,
        test_enhanced_configuration_loading,
        test_cli_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\n🔍 Running {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All enhanced framework tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
