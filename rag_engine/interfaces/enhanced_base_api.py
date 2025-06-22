"""
Enhanced base API with configuration-driven customization.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging


class AuthMethod(Enum):
    """Supported authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class RateLimitType(Enum):
    """Rate limiting strategies."""
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


@dataclass
class APICustomization:
    """Configuration for API framework customization."""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    reload: bool = False
    
    # API Behavior
    enable_docs: bool = True
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    custom_headers: Dict[str, str] = None
    
    # Security
    auth_method: AuthMethod = AuthMethod.NONE
    api_keys: List[str] = None
    jwt_secret: str = None
    cors_origins: List[str] = None
    
    # Rate Limiting
    enable_rate_limiting: bool = False
    rate_limit_type: RateLimitType = RateLimitType.PER_IP
    requests_per_minute: int = 60
    burst_size: int = 10
    
    # Middleware
    enable_compression: bool = True
    enable_request_logging: bool = True
    enable_response_caching: bool = False
    cache_ttl: int = 300
    
    # Error Handling
    include_error_details: bool = False
    custom_error_handlers: Dict[int, Callable] = None
    
    # Monitoring
    enable_metrics: bool = False
    metrics_endpoint: str = "/metrics"
    enable_health_checks: bool = True
    
    # Custom Endpoints
    custom_routes: List[Dict[str, Any]] = None
    middleware_stack: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.custom_headers is None:
            self.custom_headers = {}
        if self.api_keys is None:
            self.api_keys = []
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.custom_error_handlers is None:
            self.custom_error_handlers = {}
        if self.custom_routes is None:
            self.custom_routes = []
        if self.middleware_stack is None:
            self.middleware_stack = []


class EnhancedBaseAPIServer(ABC):
    """Enhanced base class with full customization support."""
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[Any] = None,
                 api_config: Optional[APICustomization] = None):
        """Initialize with enhanced configuration."""
        self.config = config
        self.api_config = api_config or APICustomization()
        self.middleware_handlers = {}
        self.custom_validators = {}
        self.request_interceptors = []
        self.response_transformers = []
        
    @abstractmethod
    def create_app(self) -> Any:
        """Create the framework-specific application."""
        pass
    
    @abstractmethod
    def add_middleware(self, middleware_type: str, handler: Callable) -> None:
        """Add middleware to the application."""
        pass
    
    @abstractmethod
    def add_route(self, path: str, handler: Callable, methods: List[str], **kwargs) -> None:
        """Add a custom route to the application."""
        pass
    
    @abstractmethod
    def set_error_handler(self, status_code: int, handler: Callable) -> None:
        """Set custom error handler."""
        pass
    
    @abstractmethod
    def enable_authentication(self) -> None:
        """Enable authentication based on configuration."""
        pass
    
    @abstractmethod
    def enable_rate_limiting(self) -> None:
        """Enable rate limiting based on configuration."""
        pass
    
    @abstractmethod
    def start_server(self, **kwargs) -> None:
        """Start the server with production configuration."""
        pass
    
    def register_middleware(self, name: str, handler: Callable) -> None:
        """Register custom middleware."""
        self.middleware_handlers[name] = handler
    
    def register_validator(self, endpoint: str, validator: Callable) -> None:
        """Register custom request validator."""
        self.custom_validators[endpoint] = validator
    
    def add_request_interceptor(self, interceptor: Callable) -> None:
        """Add request interceptor."""
        self.request_interceptors.append(interceptor)
    
    def add_response_transformer(self, transformer: Callable) -> None:
        """Add response transformer."""
        self.response_transformers.append(transformer)


class APIFrameworkFactory:
    """Enhanced factory with plugin support and custom server registration."""
    
    def __init__(self):
        self._frameworks = {}
        self._plugins = {}
        self._middleware_registry = {}
        self._custom_servers = {}
    
    def register_framework(self, name: str, framework_class: type, 
                          config_schema: Optional[type] = None) -> None:
        """Register a framework with optional config schema."""
        self._frameworks[name] = {
            'class': framework_class,
            'config_schema': config_schema,
            'type': 'builtin'
        }
    
    def register_custom_server(self, name: str, server_class: type, 
                              description: str = None, **default_kwargs) -> None:
        """Register a custom user-defined server."""
        self._custom_servers[name] = {
            'class': server_class,
            'description': description or f"Custom server: {name}",
            'default_kwargs': default_kwargs,
            'type': 'custom'
        }
        
        # Also register in main frameworks dict for unified access
        self._frameworks[name] = {
            'class': server_class,
            'config_schema': None,
            'type': 'custom'
        }
    
    def register_plugin(self, name: str, plugin_class: type) -> None:
        """Register a plugin that can be used across frameworks."""
        self._plugins[name] = plugin_class
    
    def register_middleware(self, name: str, middleware_class: type) -> None:
        """Register middleware that can be used across frameworks."""
        self._middleware_registry[name] = middleware_class
    
    def create_server(self, framework: str, config: Any = None, 
                     api_config: APICustomization = None, 
                     plugins: List[str] = None,
                     **custom_kwargs) -> EnhancedBaseAPIServer:
        """Create server with plugins and customization."""
        if framework not in self._frameworks:
            raise ValueError(f"Framework '{framework}' not registered. Available: {self.list_frameworks()}")
        
        framework_info = self._frameworks[framework]
        
        # Handle custom servers differently
        if framework_info['type'] == 'custom':
            # Use wrapper for custom servers
            custom_server_info = self._custom_servers.get(framework, {})
            merged_kwargs = {**custom_server_info.get('default_kwargs', {}), **custom_kwargs}
            
            server = CustomServerWrapper(
                custom_server_class=framework_info['class'],
                config=config,
                api_config=api_config,
                **merged_kwargs
            )
        else:
            # Use built-in servers
            server = framework_info['class'](config=config, api_config=api_config)
        
        # Apply plugins
        if plugins:
            for plugin_name in plugins:
                if plugin_name in self._plugins:
                    plugin = self._plugins[plugin_name](server)
                    plugin.apply()
        
        return server
    
    def list_frameworks(self) -> List[str]:
        """List available frameworks."""
        return list(self._frameworks.keys())
    
    def list_builtin_frameworks(self) -> List[str]:
        """List only built-in frameworks."""
        return [name for name, info in self._frameworks.items() if info['type'] == 'builtin']
    
    def list_custom_servers(self) -> List[str]:
        """List only custom servers."""
        return list(self._custom_servers.keys())
    
    def get_framework_info(self, framework: str) -> Dict[str, Any]:
        """Get information about a framework."""
        if framework not in self._frameworks:
            return None
        
        info = self._frameworks[framework].copy()
        if framework in self._custom_servers:
            info.update(self._custom_servers[framework])
        
        return info
    
    def list_plugins(self) -> List[str]:
        """List available plugins."""
        return list(self._plugins.keys())
    
    def list_middleware(self) -> List[str]:
        """List available middleware."""
        return list(self._middleware_registry.keys())
    
    def validate_custom_server(self, server_class: type) -> bool:
        """Validate that a custom server implements required methods."""
        required_methods = ['create_app', 'start_server']
        
        for method in required_methods:
            if not hasattr(server_class, method):
                raise ValueError(f"Custom server must implement '{method}' method")
        
        return True


# Global factory instance
enhanced_factory = APIFrameworkFactory()


class CustomServerInterface:
    """Interface that custom user servers must implement to integrate with RAG Engine."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        """Initialize custom server with RAG configuration."""
        pass
    
    def create_app(self) -> Any:
        """Create the custom application instance."""
        raise NotImplementedError("Custom server must implement create_app()")
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the custom server."""
        raise NotImplementedError("Custom server must implement start_server()")
    
    def add_rag_integration(self, rag_config: Any) -> None:
        """Optional method to integrate RAG functionality."""
        pass


class CustomServerWrapper(EnhancedBaseAPIServer):
    """Wrapper for user-defined custom servers."""
    
    def __init__(self, custom_server_class: type, 
                 config_path: Optional[str] = None, 
                 config: Optional[Any] = None,
                 api_config: Optional[APICustomization] = None,
                 **custom_kwargs):
        """Initialize wrapper with custom server class."""
        super().__init__(config_path, config, api_config)
        self.custom_server_class = custom_server_class
        self.custom_kwargs = custom_kwargs
        self.custom_server = None
        
    def create_app(self) -> Any:
        """Create the custom server application."""
        try:
            # Instantiate custom server
            self.custom_server = self.custom_server_class(
                config=self.config,
                **self.custom_kwargs
            )
            
            # Try to integrate RAG functionality if supported
            if hasattr(self.custom_server, 'add_rag_integration'):
                self.custom_server.add_rag_integration(self.config)
            
            # Create the app
            return self.custom_server.create_app()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create custom server: {e}")
    
    def start_server(self, **kwargs) -> None:
        """Start the custom server."""
        if self.custom_server is None:
            self.create_app()
        
        # Merge configuration
        server_config = {
            "host": self.api_config.host,
            "port": self.api_config.port,
            **kwargs
        }
        
        self.custom_server.start_server(**server_config)
    
    # Implement abstract methods (mostly delegate to custom server)
    def add_middleware(self, middleware_type: str, handler: Callable) -> None:
        """Add middleware (if supported by custom server)."""
        if hasattr(self.custom_server, 'add_middleware'):
            self.custom_server.add_middleware(middleware_type, handler)
        else:
            print(f"⚠️  Custom server doesn't support middleware: {middleware_type}")
    
    def add_route(self, path: str, handler: Callable, methods: List[str], **kwargs) -> None:
        """Add route (if supported by custom server)."""
        if hasattr(self.custom_server, 'add_route'):
            self.custom_server.add_route(path, handler, methods, **kwargs)
        else:
            print(f"⚠️  Custom server doesn't support dynamic routes: {path}")
    
    def set_error_handler(self, status_code: int, handler: Callable) -> None:
        """Set error handler (if supported by custom server)."""
        if hasattr(self.custom_server, 'set_error_handler'):
            self.custom_server.set_error_handler(status_code, handler)
        else:
            print(f"⚠️  Custom server doesn't support error handlers: {status_code}")
    
    def enable_authentication(self) -> None:
        """Enable authentication (if supported by custom server)."""
        if hasattr(self.custom_server, 'enable_authentication'):
            self.custom_server.enable_authentication()
    
    def enable_rate_limiting(self) -> None:
        """Enable rate limiting (if supported by custom server)."""
        if hasattr(self.custom_server, 'enable_rate_limiting'):
            self.custom_server.enable_rate_limiting()
