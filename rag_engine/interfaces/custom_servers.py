"""
Custom server support for RAG Engine.
This module provides utilities and examples for creating custom servers.
"""
from typing import Dict, Any, Optional, List, Callable
import logging
from abc import ABC, abstractmethod

from .enhanced_base_api import CustomServerInterface, enhanced_factory


class CustomServerBase(CustomServerInterface):
    """Base class to make it easier for users to create custom servers."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        """Initialize custom server with RAG configuration."""
        self.config = config
        self.rag_endpoints = {}
        self.custom_kwargs = kwargs
        self.logger = logging.getLogger(f"custom_server_{self.__class__.__name__}")
        
    def add_rag_integration(self, rag_config: Any) -> None:
        """Add RAG integration - can be overridden by users."""
        self.config = rag_config
        self.setup_rag_endpoints()
    
    def setup_rag_endpoints(self) -> None:
        """Setup standard RAG endpoints - can be overridden."""
        self.rag_endpoints = {
            'chat': self.default_chat_handler,
            'build': self.default_build_handler,
            'status': self.default_status_handler,
            'health': self.default_health_handler
        }
    
    def default_chat_handler(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Default chat handler - users can override."""
        try:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                return {"error": "Pipeline not built", "status": "error"}
            
            response = self.pipeline.query(query)
            return {
                "query": query,
                "response": response,
                "session_id": session_id,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return {"error": str(e), "status": "error"}
    
    def default_build_handler(self) -> Dict[str, Any]:
        """Default build handler - users can override."""
        try:
            from ..core.pipeline import Pipeline
            
            if not self.config:
                return {"error": "No configuration provided", "status": "error"}
            
            self.pipeline = Pipeline(self.config)
            self.pipeline.build()
            
            return {
                "message": "Pipeline built successfully",
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Build error: {e}")
            return {"error": str(e), "status": "error"}
    
    def default_status_handler(self) -> Dict[str, Any]:
        """Default status handler - users can override."""
        return {
            "status": "healthy",
            "pipeline_built": hasattr(self, 'pipeline') and self.pipeline is not None,
            "server_type": "custom",
            "server_class": self.__class__.__name__
        }
    
    def default_health_handler(self) -> Dict[str, Any]:
        """Default health handler - users can override."""
        return {
            "status": "healthy",
            "timestamp": __import__('time').time(),
            "server": self.__class__.__name__
        }


# Example implementations for popular frameworks

class TornadoCustomServer(CustomServerBase):
    """Example custom server using Tornado."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        try:
            import tornado.web
            import tornado.ioloop
            self.tornado = tornado
            self.app = None
        except ImportError:
            raise ImportError("Tornado not installed. Install with: pip install tornado")
    
    def create_app(self):
        """Create Tornado application."""
        import tornado.web
        import json
        
        class BaseHandler(tornado.web.RequestHandler):
            def initialize(self, server):
                self.server = server
        
        class ChatHandler(BaseHandler):
            def post(self):
                try:
                    data = json.loads(self.request.body)
                    query = data.get('query')
                    session_id = data.get('session_id')
                    
                    result = self.server.rag_endpoints['chat'](query, session_id)
                    self.write(result)
                except Exception as e:
                    self.set_status(500)
                    self.write({"error": str(e), "status": "error"})
        
        class BuildHandler(BaseHandler):
            def post(self):
                try:
                    result = self.server.rag_endpoints['build']()
                    self.write(result)
                except Exception as e:
                    self.set_status(500)
                    self.write({"error": str(e), "status": "error"})
        
        class StatusHandler(BaseHandler):
            def get(self):
                result = self.server.rag_endpoints['status']()
                self.write(result)
        
        class HealthHandler(BaseHandler):
            def get(self):
                result = self.server.rag_endpoints['health']()
                self.write(result)
        
        self.app = tornado.web.Application([
            (r"/chat", ChatHandler, dict(server=self)),
            (r"/build", BuildHandler, dict(server=self)),
            (r"/status", StatusHandler, dict(server=self)),
            (r"/health", HealthHandler, dict(server=self)),
        ])
        
        return self.app
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start Tornado server."""
        if not self.app:
            self.create_app()
        
        print(f"🌪️  Starting Tornado custom server on http://{host}:{port}")
        self.app.listen(port, address=host)
        self.tornado.ioloop.IOLoop.current().start()


class BottleCustomServer(CustomServerBase):
    """Example custom server using Bottle."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        try:
            import bottle
            self.bottle = bottle
            self.app = bottle.Bottle()
        except ImportError:
            raise ImportError("Bottle not installed. Install with: pip install bottle")
    
    def create_app(self):
        """Create Bottle application."""
        import json
        
        @self.app.post('/chat')
        def chat():
            try:
                data = self.bottle.request.json
                query = data.get('query')
                session_id = data.get('session_id')
                
                result = self.rag_endpoints['chat'](query, session_id)
                return result
            except Exception as e:
                self.bottle.response.status = 500
                return {"error": str(e), "status": "error"}
        
        @self.app.post('/build')
        def build():
            try:
                result = self.rag_endpoints['build']()
                return result
            except Exception as e:
                self.bottle.response.status = 500
                return {"error": str(e), "status": "error"}
        
        @self.app.get('/status')
        def status():
            return self.rag_endpoints['status']()
        
        @self.app.get('/health')
        def health():
            return self.rag_endpoints['health']()
        
        return self.app
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start Bottle server."""
        if not self.app:
            self.create_app()
        
        print(f"🍼 Starting Bottle custom server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=kwargs.get('debug', False))


class CherryPyCustomServer(CustomServerBase):
    """Example custom server using CherryPy."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        try:
            import cherrypy
            self.cherrypy = cherrypy
        except ImportError:
            raise ImportError("CherryPy not installed. Install with: pip install cherrypy")
    
    def create_app(self):
        """Create CherryPy application."""
        import cherrypy
        import json
        
        class RAGAPIServer:
            def __init__(self, rag_server):
                self.rag_server = rag_server
            
            @cherrypy.expose
            @cherrypy.tools.json_in()
            @cherrypy.tools.json_out()
            def chat(self):
                try:
                    data = cherrypy.request.json
                    query = data.get('query')
                    session_id = data.get('session_id')
                    
                    return self.rag_server.rag_endpoints['chat'](query, session_id)
                except Exception as e:
                    cherrypy.response.status = 500
                    return {"error": str(e), "status": "error"}
            
            @cherrypy.expose
            @cherrypy.tools.json_out()
            def build(self):
                try:
                    return self.rag_server.rag_endpoints['build']()
                except Exception as e:
                    cherrypy.response.status = 500
                    return {"error": str(e), "status": "error"}
            
            @cherrypy.expose
            @cherrypy.tools.json_out()
            def status(self):
                return self.rag_server.rag_endpoints['status']()
            
            @cherrypy.expose
            @cherrypy.tools.json_out()
            def health(self):
                return self.rag_server.rag_endpoints['health']()
        
        self.app = RAGAPIServer(self)
        return self.app
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start CherryPy server."""
        if not self.app:
            self.create_app()
        
        print(f"🍒 Starting CherryPy custom server on http://{host}:{port}")
        
        cherrypy.config.update({
            'server.socket_host': host,
            'server.socket_port': port,
            'engine.autoreload.on': kwargs.get('debug', False)
        })
        
        cherrypy.quickstart(self.app)


def register_example_servers():
    """Register example custom servers with the factory."""
    try:
        enhanced_factory.register_custom_server(
            "tornado",
            TornadoCustomServer,
            "High-performance async server using Tornado"
        )
    except ImportError:
        pass
    
    try:
        enhanced_factory.register_custom_server(
            "bottle",
            BottleCustomServer,
            "Lightweight WSGI server using Bottle"
        )
    except ImportError:
        pass
    
    try:
        enhanced_factory.register_custom_server(
            "cherrypy",
            CherryPyCustomServer,
            "Object-oriented server using CherryPy"
        )
    except ImportError:
        pass


# Helper functions for users

def create_custom_server_template(name: str, framework: str = "custom") -> str:
    """Generate a template for creating custom servers."""
    
    template = '''"""
Custom RAG Engine server: {name}
"""
from typing import Dict, Any, Optional
from rag_engine.interfaces.custom_servers import CustomServerBase


class {name_title}Server(CustomServerBase):
    """Custom RAG Engine server using {framework}."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        # Initialize your framework here
        # Example: self.framework_app = {framework}.create_app()
    
    def create_app(self):
        """Create your application instance."""
        # Implement your framework-specific app creation
        # Make sure to setup routes for RAG endpoints:
        # - POST /chat
        # - POST /build  
        # - GET /status
        # - GET /health
        
        # You can use self.rag_endpoints to get default handlers:
        # chat_result = self.rag_endpoints['chat'](query, session_id)
        # build_result = self.rag_endpoints['build']()
        # status_result = self.rag_endpoints['status']()
        # health_result = self.rag_endpoints['health']()
        
        pass  # Replace with your implementation
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start your server."""
        if not hasattr(self, 'app') or self.app is None:
            self.create_app()
        
        print(f"🚀 Starting {name} custom server on {{host}}:{{port}}")
        
        # Start your server here
        # Example: self.framework_app.run(host=host, port=port)
        pass  # Replace with your implementation


# Register your custom server
from rag_engine.interfaces.enhanced_base_api import enhanced_factory

enhanced_factory.register_custom_server(
    "{name_lower}",
    {name_title}Server,
    "Custom server using {framework}"
)
'''
    
    return template.format(
        name=name,
        name_title=name.title(),
        name_lower=name.lower(),
        framework=framework
    )


def validate_custom_server_implementation(server_class: type) -> List[str]:
    """Validate a custom server implementation and return any issues."""
    issues = []
    
    required_methods = ['create_app', 'start_server']
    for method in required_methods:
        if not hasattr(server_class, method):
            issues.append(f"Missing required method: {method}")
    
    # Check if it inherits from a base class
    if not any(base.__name__ in ['CustomServerInterface', 'CustomServerBase'] 
              for base in server_class.__mro__):
        issues.append("Server should inherit from CustomServerInterface or CustomServerBase")
    
    return issues
