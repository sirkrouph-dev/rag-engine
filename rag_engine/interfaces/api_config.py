"""
Configuration schema for customizable API frameworks.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class FrameworkType(str, Enum):
    """Supported API frameworks."""
    FASTAPI = "fastapi"
    FLASK = "flask"
    DJANGO = "django"


class AuthMethod(str, Enum):
    """Authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CORSConfig(BaseModel):
    """CORS configuration."""
    enabled: bool = True
    origins: List[str] = ["*"]
    methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    headers: List[str] = ["*"]
    credentials: bool = True


class SecurityConfig(BaseModel):
    """Security configuration."""
    enable_auth: bool = False
    auth_method: AuthMethod = AuthMethod.NONE
    api_keys: List[str] = []
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # IP filtering
    allowed_ips: List[str] = []
    blocked_ips: List[str] = []
    
    # Security headers
    enable_security_headers: bool = True
    custom_headers: Dict[str, str] = {}


class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = False
    backend: str = "memory"  # memory, redis, memcached
    ttl_seconds: int = 300
    max_size: int = 1000
    redis_url: Optional[str] = None


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    enable_prometheus: bool = False
    enable_health_checks: bool = True
    health_endpoint: str = "/health"
    
    # Performance tracking
    track_response_times: bool = True
    track_request_counts: bool = True
    track_error_rates: bool = True
    
    # Alerting
    enable_alerting: bool = False
    response_time_threshold: float = 5.0
    error_rate_threshold: float = 0.05


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json, text
    enable_request_logging: bool = True
    enable_response_logging: bool = True
    include_request_body: bool = False
    include_response_body: bool = False
    log_file: Optional[str] = None


class CustomRoute(BaseModel):
    """Custom route definition."""
    path: str
    method: str = "GET"
    handler: str  # Import path to handler function
    middleware: List[str] = []
    auth_required: bool = False


class PluginConfig(BaseModel):
    """Plugin configuration."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = {}


class APICustomizationConfig(BaseModel):
    """Complete API customization configuration."""
    
    # Framework selection
    framework: FrameworkType = FrameworkType.FASTAPI
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    reload: bool = False
    
    # API documentation
    enable_docs: bool = True
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    title: str = "RAG Engine API"
    description: str = "A modular Retrieval-Augmented Generation framework API"
    version: str = "1.0.0"
    
    # Cross-cutting concerns
    cors: CORSConfig = CORSConfig()
    security: SecurityConfig = SecurityConfig()
    caching: CacheConfig = CacheConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Custom extensions
    custom_routes: List[CustomRoute] = []
    plugins: List[PluginConfig] = []
    middleware_stack: List[str] = []
    
    # Framework-specific settings
    framework_specific: Dict[str, Any] = {}


class APIConfigurationManager:
    """Manages API configuration loading and validation."""
    
    @staticmethod
    def load_from_file(config_path: str) -> APICustomizationConfig:
        """Load configuration from file."""
        import json
        import yaml
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                data = json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config format")
        
        return APICustomizationConfig(**data)
    
    @staticmethod
    def load_from_env() -> APICustomizationConfig:
        """Load configuration from environment variables."""
        import os
        
        config_data = {}
        
        # Map environment variables to config
        env_mappings = {
            'RAG_API_FRAMEWORK': 'framework',
            'RAG_API_HOST': 'host',
            'RAG_API_PORT': 'port',
            'RAG_API_WORKERS': 'workers',
            'RAG_API_DEBUG': 'debug',
            'RAG_API_ENABLE_AUTH': 'security.enable_auth',
            'RAG_API_AUTH_METHOD': 'security.auth_method',
            'RAG_API_JWT_SECRET': 'security.jwt_secret',
            'RAG_API_RATE_LIMIT': 'security.rate_limit_per_minute',
            'RAG_API_ENABLE_METRICS': 'monitoring.enable_metrics',
            'RAG_API_LOG_LEVEL': 'logging.level'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle nested config paths
                keys = config_path.split('.')
                current = config_data
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert types
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                
                current[keys[-1]] = value
        
        return APICustomizationConfig(**config_data)
    
    @staticmethod
    def merge_configs(base: APICustomizationConfig, 
                     override: APICustomizationConfig) -> APICustomizationConfig:
        """Merge two configurations."""
        base_dict = base.dict()
        override_dict = override.dict()
        
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
        
        deep_merge(base_dict, override_dict)
        return APICustomizationConfig(**base_dict)


# Example configurations

DEVELOPMENT_CONFIG = APICustomizationConfig(
    debug=True,
    reload=True,
    workers=1,
    security=SecurityConfig(
        enable_auth=False,
        enable_rate_limiting=False
    ),
    logging=LoggingConfig(
        level=LogLevel.DEBUG,
        enable_request_logging=True
    ),
    monitoring=MonitoringConfig(
        enable_metrics=True,
        enable_prometheus=False
    )
)

PRODUCTION_CONFIG = APICustomizationConfig(
    debug=False,
    reload=False,
    workers=4,
    security=SecurityConfig(
        enable_auth=True,
        auth_method=AuthMethod.API_KEY,
        enable_rate_limiting=True,
        rate_limit_per_minute=1000,
        enable_security_headers=True
    ),
    caching=CacheConfig(
        enabled=True,
        backend="redis",
        ttl_seconds=300
    ),
    logging=LoggingConfig(
        level=LogLevel.INFO,
        format="json",
        enable_request_logging=True
    ),
    monitoring=MonitoringConfig(
        enable_metrics=True,
        enable_prometheus=True,
        enable_alerting=True
    ),
    plugins=[
        PluginConfig(name="cache", enabled=True),
        PluginConfig(name="logging", enabled=True),
        PluginConfig(name="validation", enabled=True)
    ]
)

MICROSERVICE_CONFIG = APICustomizationConfig(
    framework=FrameworkType.FASTAPI,
    workers=2,
    security=SecurityConfig(
        enable_auth=True,
        auth_method=AuthMethod.JWT,
        enable_rate_limiting=True
    ),
    monitoring=MonitoringConfig(
        enable_metrics=True,
        enable_prometheus=True,
        enable_health_checks=True
    ),
    custom_routes=[
        CustomRoute(
            path="/custom/endpoint",
            method="POST",
            handler="my_module.custom_handler",
            auth_required=True
        )
    ]
)
