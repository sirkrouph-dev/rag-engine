"""
Plugin system for extending API framework functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Type
import importlib
import inspect
from dataclasses import dataclass


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    supported_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.supported_frameworks is None:
            self.supported_frameworks = ["fastapi", "flask", "django"]


class APIPlugin(ABC):
    """Base class for API plugins."""
    
    def __init__(self, api_server):
        self.api_server = api_server
        self.metadata = self.get_metadata()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def apply(self) -> None:
        """Apply plugin functionality to the API server."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """Return additional routes provided by this plugin."""
        return []
    
    def get_middleware(self) -> List[Callable]:
        """Return middleware provided by this plugin."""
        return []
    
    def get_dependencies(self) -> List[str]:
        """Return plugin dependencies."""
        return self.metadata.dependencies


class CachePlugin(APIPlugin):
    """Response caching plugin."""
    
    def __init__(self, api_server, cache_ttl: int = 300, cache_size: int = 1000):
        super().__init__(api_server)
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self.cache = {}
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="cache",
            version="1.0.0",
            description="Response caching plugin",
            author="RAG Engine Team"
        )
    
    def initialize(self) -> bool:
        """Initialize cache storage."""
        try:
            # Could initialize Redis or other cache backends here
            return True
        except Exception:
            return False
    
    def apply(self) -> None:
        """Apply caching middleware."""
        cache_middleware = self._create_cache_middleware()
        self.api_server.add_middleware("cache", cache_middleware)
    
    def _create_cache_middleware(self) -> Callable:
        """Create caching middleware."""
        def cache_middleware(request, call_next):
            # Simple in-memory caching logic
            cache_key = f"{request.method}:{request.url}"
            
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_response
            
            response = call_next(request)
            
            # Cache GET requests only
            if request.method == "GET" and response.status_code == 200:
                self.cache[cache_key] = (response, time.time())
                
                # Simple LRU eviction
                if len(self.cache) > self.cache_size:
                    oldest_key = min(self.cache.keys(), 
                                   key=lambda k: self.cache[k][1])
                    del self.cache[oldest_key]
            
            return response
        
        return cache_middleware


class LoggingPlugin(APIPlugin):
    """Enhanced logging plugin."""
    
    def __init__(self, api_server, log_level: str = "INFO", 
                 log_format: str = "json", include_body: bool = False):
        super().__init__(api_server)
        self.log_level = log_level
        self.log_format = log_format
        self.include_body = include_body
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging",
            version="1.0.0",
            description="Enhanced request/response logging",
            author="RAG Engine Team"
        )
    
    def initialize(self) -> bool:
        """Initialize logging configuration."""
        import logging
        import json
        
        # Setup structured logging
        self.logger = logging.getLogger("api_requests")
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        return True
    
    def apply(self) -> None:
        """Apply logging middleware."""
        logging_middleware = self._create_logging_middleware()
        self.api_server.add_middleware("logging", logging_middleware)
    
    def _create_logging_middleware(self) -> Callable:
        """Create logging middleware."""
        def logging_middleware(request, call_next):
            import time
            import json
            
            start_time = time.time()
            
            # Log request
            request_data = {
                "timestamp": time.time(),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host
            }
            
            if self.include_body and request.method in ["POST", "PUT", "PATCH"]:
                # Note: This would need to handle async body reading properly
                pass
            
            self.logger.info(f"Request: {json.dumps(request_data)}")
            
            # Process request
            response = call_next(request)
            
            # Log response
            response_time = time.time() - start_time
            response_data = {
                "timestamp": time.time(),
                "status_code": response.status_code,
                "response_time": response_time,
                "content_length": response.headers.get("content-length")
            }
            
            self.logger.info(f"Response: {json.dumps(response_data)}")
            
            return response
        
        return logging_middleware


class ValidationPlugin(APIPlugin):
    """Request/response validation plugin."""
    
    def __init__(self, api_server, schemas: Dict[str, Any] = None):
        super().__init__(api_server)
        self.schemas = schemas or {}
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="validation",
            version="1.0.0",
            description="Request/response validation",
            author="RAG Engine Team",
            dependencies=["jsonschema"]
        )
    
    def initialize(self) -> bool:
        """Initialize validation schemas."""
        try:
            import jsonschema
            self.validator = jsonschema
            return True
        except ImportError:
            return False
    
    def apply(self) -> None:
        """Apply validation middleware."""
        validation_middleware = self._create_validation_middleware()
        self.api_server.add_middleware("validation", validation_middleware)
    
    def _create_validation_middleware(self) -> Callable:
        """Create validation middleware."""
        def validation_middleware(request, call_next):
            # Validate request based on endpoint schema
            endpoint = request.url.path
            if endpoint in self.schemas:
                schema = self.schemas[endpoint]
                # Validation logic here
                pass
            
            return call_next(request)
        
        return validation_middleware


class PluginManager:
    """Manages API plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, APIPlugin] = {}
        self.plugin_registry: Dict[str, Type[APIPlugin]] = {}
        self.loaded_plugins: List[str] = []
    
    def register_plugin(self, plugin_class: Type[APIPlugin]) -> None:
        """Register a plugin class."""
        # Get metadata from plugin
        temp_instance = plugin_class(None)
        metadata = temp_instance.get_metadata()
        self.plugin_registry[metadata.name] = plugin_class
    
    def load_plugin(self, plugin_name: str, api_server, **kwargs) -> bool:
        """Load and initialize a plugin."""
        if plugin_name not in self.plugin_registry:
            raise ValueError(f"Plugin '{plugin_name}' not registered")
        
        plugin_class = self.plugin_registry[plugin_name]
        plugin_instance = plugin_class(api_server, **kwargs)
        
        # Check dependencies
        for dep in plugin_instance.get_dependencies():
            if dep not in self.loaded_plugins:
                # Try to load dependency
                if not self.load_plugin(dep, api_server):
                    return False
        
        # Initialize plugin
        if not plugin_instance.initialize():
            return False
        
        # Apply plugin
        plugin_instance.apply()
        
        self.plugins[plugin_name] = plugin_instance
        self.loaded_plugins.append(plugin_name)
        
        return True
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        plugin.cleanup()
        
        del self.plugins[plugin_name]
        self.loaded_plugins.remove(plugin_name)
        
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[APIPlugin]:
        """Get a loaded plugin."""
        return self.plugins.get(plugin_name)
    
    def list_available_plugins(self) -> List[str]:
        """List available plugins."""
        return list(self.plugin_registry.keys())
    
    def list_loaded_plugins(self) -> List[str]:
        """List loaded plugins."""
        return self.loaded_plugins.copy()
    
    def auto_discover_plugins(self, package_name: str = "rag_engine.plugins") -> None:
        """Auto-discover plugins in a package."""
        try:
            package = importlib.import_module(package_name)
            for item_name in dir(package):
                item = getattr(package, item_name)
                if (inspect.isclass(item) and 
                    issubclass(item, APIPlugin) and 
                    item != APIPlugin):
                    self.register_plugin(item)
        except ImportError:
            pass


# Global plugin manager
plugin_manager = PluginManager()

# Register built-in plugins
plugin_manager.register_plugin(CachePlugin)
plugin_manager.register_plugin(LoggingPlugin)
plugin_manager.register_plugin(ValidationPlugin)
