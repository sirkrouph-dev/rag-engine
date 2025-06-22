"""
Enhanced Flask implementation with full customization support.
"""
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from functools import wraps
from typing import Optional, Dict, Any, List, Callable
import time
import logging
from collections import defaultdict, deque
import json

from .enhanced_base_api import EnhancedBaseAPIServer, APICustomization, AuthMethod, RateLimitType
from .monitoring import MetricsCollector, HealthChecker


class FlaskEnhanced(EnhancedBaseAPIServer):
    """Enhanced Flask implementation with full customization support."""
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[Any] = None,
                 api_config: Optional[APICustomization] = None):
        super().__init__(config_path, config, api_config)
        self.app = None
        self.request_counts = defaultdict(lambda: deque())
        self.cache = {}
        self.cache_times = {}
        self.metrics = MetricsCollector() if api_config and api_config.enable_metrics else None
        self.health_checker = HealthChecker() if api_config and api_config.enable_health_checks else None
        
    def create_app(self) -> Flask:
        """Create and configure the enhanced Flask application."""
        if self.app is not None:
            return self.app
            
        self.app = Flask(__name__)
        
        # Basic configuration
        self.app.config['DEBUG'] = self.api_config.debug
        
        # Apply customizations
        self._setup_cors()
        self._setup_middleware()
        self._setup_authentication()
        self._setup_rate_limiting()
        self._setup_error_handlers()
        self._setup_core_routes()
        self._setup_custom_routes()
        self._setup_monitoring()
        
        return self.app
    
    def _setup_cors(self):
        """Setup CORS based on configuration."""
        if self.api_config.cors_origins:
            CORS(
                self.app,
                origins=self.api_config.cors_origins,
                allow_headers=["*"],
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            )
    
    def _setup_middleware(self):
        """Setup middleware based on configuration."""
        @self.app.before_request
        def before_request():
            """Request preprocessing."""
            g.start_time = time.time()
            
            # Apply request interceptors
            for interceptor in self.request_interceptors:
                interceptor(request)
        
        @self.app.after_request
        def after_request(response):
            """Response postprocessing."""
            # Add custom headers
            for header, value in self.api_config.custom_headers.items():
                response.headers[header] = value
            
            # Request logging
            if self.api_config.enable_request_logging:
                duration = time.time() - g.start_time
                logging.info(
                    f"{request.method} {request.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {duration:.4f}s"
                )
                
                if self.metrics:
                    self.metrics.record_request(
                        method=request.method,
                        endpoint=request.path,
                        status_code=response.status_code,
                        duration=duration
                    )
            
            # Apply response transformers
            for transformer in self.response_transformers:
                response = transformer(response)
            
            return response
        
        # Compression middleware (basic implementation)
        if self.api_config.enable_compression:
            @self.app.after_request
            def compress_response(response):
                if (response.content_length and 
                    response.content_length > 1000 and
                    'gzip' in request.headers.get('Accept-Encoding', '')):
                    import gzip
                    response.data = gzip.compress(response.data)
                    response.headers['Content-Encoding'] = 'gzip'
                    response.headers['Content-Length'] = len(response.data)
                return response
    
    def _setup_authentication(self):
        """Setup authentication decorators."""
        if self.api_config.auth_method == AuthMethod.NONE:
            self.auth_required = lambda f: f  # No-op decorator
            return
        
        def auth_required(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if self.api_config.auth_method == AuthMethod.API_KEY:
                    api_key = request.headers.get('X-API-Key')
                    if not api_key or api_key not in self.api_config.api_keys:
                        return jsonify({"error": "Invalid API key"}), 401
                
                elif self.api_config.auth_method == AuthMethod.JWT:
                    auth_header = request.headers.get('Authorization')
                    if not auth_header or not auth_header.startswith('Bearer '):
                        return jsonify({"error": "Missing or invalid token"}), 401
                    
                    token = auth_header.split(' ')[1]
                    try:
                        import jwt
                        payload = jwt.decode(
                            token,
                            self.api_config.jwt_secret,
                            algorithms=["HS256"]
                        )
                        g.user = payload
                    except jwt.InvalidTokenError:
                        return jsonify({"error": "Invalid token"}), 401
                
                return f(*args, **kwargs)
            return decorated_function
        
        self.auth_required = auth_required
    
    def _setup_rate_limiting(self):
        """Setup rate limiting middleware."""
        if not self.api_config.enable_rate_limiting:
            return
        
        @self.app.before_request
        def rate_limit():
            """Apply rate limiting."""
            # Determine rate limit key
            if self.api_config.rate_limit_type == RateLimitType.PER_IP:
                key = request.remote_addr
            elif self.api_config.rate_limit_type == RateLimitType.PER_ENDPOINT:
                key = f"{request.remote_addr}:{request.path}"
            else:
                key = "global"
            
            # Clean old requests
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            while self.request_counts[key] and self.request_counts[key][0] < window_start:
                self.request_counts[key].popleft()
            
            # Check rate limit
            if len(self.request_counts[key]) >= self.api_config.requests_per_minute:
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Record request
            self.request_counts[key].append(now)
    
    def _setup_error_handlers(self):
        """Setup custom error handlers."""
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                "error": "Not Found",
                "message": "The requested resource was not found",
                "path": request.path
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logging.error(f"Internal server error: {error}")
            content = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
            
            if self.api_config.include_error_details:
                content["details"] = str(error)
            
            return jsonify(content), 500
        
        # Apply custom error handlers
        for status_code, handler in self.api_config.custom_error_handlers.items():
            self.app.errorhandler(status_code)(handler)
    
    def _setup_core_routes(self):
        """Setup core RAG Engine API routes."""
        @self.app.route('/chat', methods=['POST'])
        @self.auth_required
        def chat_endpoint():
            """Chat with the RAG engine."""
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                return jsonify({"error": "Pipeline not built. Call /build first."}), 400
            
            try:
                data = request.get_json()
                if not data or 'query' not in data:
                    return jsonify({"error": "Query is required"}), 400
                
                # Check cache for GET-like behavior
                if self.api_config.enable_response_caching:
                    cache_key = f"chat:{data['query']}"
                    if cache_key in self.cache:
                        cache_time = self.cache_times.get(cache_key, 0)
                        if time.time() - cache_time < self.api_config.cache_ttl:
                            cached_response = self.cache[cache_key]
                            return jsonify(cached_response)
                
                result = self.pipeline.query(data['query'])
                response_data = {
                    "query": data['query'],
                    "response": result,
                    "session_id": data.get('session_id'),
                    "status": "success"
                }
                
                # Cache response
                if self.api_config.enable_response_caching:
                    self.cache[cache_key] = response_data
                    self.cache_times[cache_key] = time.time()
                
                return jsonify(response_data)
                
            except Exception as e:
                logging.error(f"Chat error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/build', methods=['POST'])
        @self.auth_required
        def build_endpoint():
            """Build the RAG pipeline."""
            try:
                from ..core.pipeline import Pipeline
                
                # Build pipeline in background (simplified for Flask)
                self.pipeline = Pipeline(config=self.config)
                self.pipeline.build()
                
                return jsonify({
                    "message": "Pipeline built successfully",
                    "status": "success"
                })
                
            except Exception as e:
                logging.error(f"Build error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def status_endpoint():
            """Get pipeline status."""
            return jsonify({
                "status": "healthy",
                "pipeline_built": hasattr(self, 'pipeline') and self.pipeline is not None,
                "config": self.config.dict() if self.config else {}
            })
    
    def _setup_custom_routes(self):
        """Setup custom routes from configuration."""
        for route_config in self.api_config.custom_routes:
            path = route_config.get("path")
            methods = route_config.get("methods", ["GET"])
            handler = route_config.get("handler")
            
            if path and handler:
                self.app.add_url_rule(
                    rule=path,
                    endpoint=f"custom_{path.replace('/', '_')}",
                    view_func=handler,
                    methods=methods
                )
    
    def _setup_monitoring(self):
        """Setup monitoring endpoints."""
        if self.api_config.enable_metrics and self.metrics:
            @self.app.route(self.api_config.metrics_endpoint, methods=['GET'])
            def metrics_endpoint():
                """Prometheus-style metrics endpoint."""
                return self.metrics.generate_prometheus_metrics(), 200, {
                    'Content-Type': 'text/plain'
                }
        
        if self.api_config.enable_health_checks and self.health_checker:
            @self.app.route('/health', methods=['GET'])
            def health_endpoint():
                """Health check endpoint."""
                health_status = self.health_checker.check_health()
                status_code = 200 if health_status["status"] == "healthy" else 503
                return jsonify(health_status), status_code
            
            @self.app.route('/health/ready', methods=['GET'])
            def readiness_endpoint():
                """Readiness check endpoint."""
                ready_status = self.health_checker.check_readiness()
                status_code = 200 if ready_status["ready"] else 503
                return jsonify(ready_status), status_code
    
    def add_middleware(self, middleware_type: str, handler: Callable) -> None:
        """Add custom middleware to the application."""
        if middleware_type == "before_request":
            self.app.before_request(handler)
        elif middleware_type == "after_request":
            self.app.after_request(handler)
        else:
            # For other middleware types, wrap as before_request
            self.app.before_request(handler)
    
    def add_route(self, path: str, handler: Callable, methods: List[str], **kwargs) -> None:
        """Add a custom route to the application."""
        self.app.add_url_rule(
            rule=path,
            endpoint=kwargs.get('endpoint', f"custom_{path.replace('/', '_')}"),
            view_func=handler,
            methods=methods
        )
    
    def set_error_handler(self, status_code: int, handler: Callable) -> None:
        """Set custom error handler."""
        self.app.errorhandler(status_code)(handler)
    
    def enable_authentication(self) -> None:
        """Enable authentication (already handled in _setup_authentication)."""
        pass
    
    def enable_rate_limiting(self) -> None:
        """Enable rate limiting (already handled in _setup_rate_limiting)."""
        pass
    
    def start_server(self, **kwargs) -> None:
        """Start the Flask server with production configuration."""
        if self.app is None:
            self.create_app()
        
        # For production, use Gunicorn
        if self.api_config.workers > 1:
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "-m", "gunicorn",
                "--workers", str(self.api_config.workers),
                "--bind", f"{self.api_config.host}:{self.api_config.port}",
                "--worker-class", "sync",
                "rag_engine.interfaces.flask_enhanced:create_production_app()"
            ]
            
            subprocess.run(cmd)
        else:
            # Development mode
            self.app.run(
                host=self.api_config.host,
                port=self.api_config.port,
                debug=self.api_config.debug,
                **kwargs
            )


def create_production_app():
    """Factory function for multi-worker deployment."""
    from ..config.loader import ConfigLoader
    
    # Load production configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/production.json")
    
    # Create API customization
    api_config = APICustomization(
        host="0.0.0.0",
        port=8000,
        workers=4,
        debug=False,
        enable_docs=False,
        enable_metrics=True,
        enable_health_checks=True,
        enable_rate_limiting=True,
        enable_compression=True,
        enable_request_logging=True,
        cors_origins=["https://yourdomain.com"],
        auth_method=AuthMethod.API_KEY,
        api_keys=["your-production-api-key"]
    )
    
    server = FlaskEnhanced(config=config, api_config=api_config)
    return server.create_app()
