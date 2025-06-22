"""
Enhanced FastAPI implementation with full customization support.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, APIKeyHeader
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Callable
import uvicorn
import time
import logging
from collections import defaultdict, deque
import asyncio

from .enhanced_base_api import EnhancedBaseAPIServer, APICustomization, AuthMethod, RateLimitType
from .monitoring import MetricsCollector, HealthChecker


class FastAPIEnhanced(EnhancedBaseAPIServer):
    """Enhanced FastAPI implementation with full customization support."""
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[Any] = None,
                 api_config: Optional[APICustomization] = None):
        super().__init__(config_path, config, api_config)
        self.app = None
        self.rate_limiters = {}
        self.metrics = MetricsCollector() if api_config and api_config.enable_metrics else None
        self.health_checker = HealthChecker() if api_config and api_config.enable_health_checks else None
        
    def create_app(self) -> FastAPI:
        """Create and configure the enhanced FastAPI application."""
        if self.app is not None:
            return self.app
            
        # Create FastAPI app with customization
        app_kwargs = {
            "title": "Enhanced RAG Engine API",
            "description": "Production-ready RAG Engine with full customization",
            "version": "2.0.0",
        }
        
        if self.api_config.enable_docs:
            app_kwargs["docs_url"] = self.api_config.docs_url
            app_kwargs["openapi_url"] = self.api_config.openapi_url
        else:
            app_kwargs["docs_url"] = None
            app_kwargs["openapi_url"] = None
            
        self.app = FastAPI(**app_kwargs)
        
        # Apply customizations
        self._setup_middleware()
        self._setup_authentication()
        self._setup_rate_limiting()
        self._setup_error_handlers()
        self._setup_core_routes()
        self._setup_custom_routes()
        self._setup_monitoring()
        
        return self.app
    
    def _setup_middleware(self):
        """Setup middleware based on configuration."""
        # CORS
        if self.api_config.cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.api_config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Compression
        if self.api_config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging
        if self.api_config.enable_request_logging:
            @self.app.middleware("http")
            async def log_requests(request: Request, call_next):
                start_time = time.time()
                response = await call_next(request)
                process_time = time.time() - start_time
                
                logging.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
                
                if self.metrics:
                    self.metrics.record_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code,
                        duration=process_time
                    )
                
                return response
        
        # Custom headers
        if self.api_config.custom_headers:
            @self.app.middleware("http")
            async def add_custom_headers(request: Request, call_next):
                response = await call_next(request)
                for header, value in self.api_config.custom_headers.items():
                    response.headers[header] = value
                return response
        
        # Response caching
        if self.api_config.enable_response_caching:
            self._setup_caching_middleware()
    
    def _setup_caching_middleware(self):
        """Setup response caching middleware."""
        cache = {}
        cache_times = {}
        
        @self.app.middleware("http")
        async def cache_responses(request: Request, call_next):
            # Only cache GET requests
            if request.method != "GET":
                return await call_next(request)
            
            cache_key = f"{request.url.path}?{request.url.query}"
            
            # Check cache
            if cache_key in cache:
                cache_time = cache_times.get(cache_key, 0)
                if time.time() - cache_time < self.api_config.cache_ttl:
                    cached_response = cache[cache_key]
                    return JSONResponse(
                        content=cached_response["content"],
                        status_code=cached_response["status_code"],
                        headers={"X-Cache": "HIT"}
                    )
            
            # Get fresh response
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                import json
                try:
                    content = json.loads(response_body.decode())
                    cache[cache_key] = {
                        "content": content,
                        "status_code": response.status_code
                    }
                    cache_times[cache_key] = time.time()
                except:
                    pass  # Don't cache if we can't parse JSON
                
                # Return new response
                return JSONResponse(
                    content=content,
                    status_code=response.status_code,
                    headers={"X-Cache": "MISS"}
                )
            
            return response
    
    def _setup_authentication(self):
        """Setup authentication based on configuration."""
        if self.api_config.auth_method == AuthMethod.NONE:
            return
        
        if self.api_config.auth_method == AuthMethod.API_KEY:
            api_key_header = APIKeyHeader(name="X-API-Key")
            
            async def verify_api_key(api_key: str = Depends(api_key_header)):
                if api_key not in self.api_config.api_keys:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                return api_key
            
            # Apply to all protected routes
            self.auth_dependency = verify_api_key
        
        elif self.api_config.auth_method == AuthMethod.JWT:
            bearer_scheme = HTTPBearer()
            
            async def verify_jwt(token: str = Depends(bearer_scheme)):
                try:
                    import jwt
                    payload = jwt.decode(
                        token.credentials,
                        self.api_config.jwt_secret,
                        algorithms=["HS256"]
                    )
                    return payload
                except jwt.InvalidTokenError:
                    raise HTTPException(status_code=401, detail="Invalid token")
            
            self.auth_dependency = verify_jwt
    
    def _setup_rate_limiting(self):
        """Setup rate limiting based on configuration."""
        if not self.api_config.enable_rate_limiting:
            return
        
        # In-memory rate limiter (for production, use Redis)
        request_counts = defaultdict(lambda: deque())
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Determine rate limit key
            if self.api_config.rate_limit_type == RateLimitType.PER_IP:
                key = request.client.host
            elif self.api_config.rate_limit_type == RateLimitType.PER_ENDPOINT:
                key = f"{request.client.host}:{request.url.path}"
            else:
                key = "global"
            
            # Clean old requests
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            while request_counts[key] and request_counts[key][0] < window_start:
                request_counts[key].popleft()
            
            # Check rate limit
            if len(request_counts[key]) >= self.api_config.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
            
            # Record request
            request_counts[key].append(now)
            
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.api_config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                self.api_config.requests_per_minute - len(request_counts[key])
            )
            
            return response
    
    def _setup_error_handlers(self):
        """Setup custom error handlers."""
        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Not Found",
                    "message": "The requested resource was not found",
                    "path": request.url.path
                }
            )
        
        @self.app.exception_handler(500)
        async def internal_error_handler(request: Request, exc: Exception):
            logging.error(f"Internal server error: {exc}")
            content = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
            
            if self.api_config.include_error_details:
                content["details"] = str(exc)
            
            return JSONResponse(status_code=500, content=content)
        
        # Apply custom error handlers
        for status_code, handler in self.api_config.custom_error_handlers.items():
            self.app.exception_handler(status_code)(handler)
    
    def _setup_core_routes(self):
        """Setup core RAG Engine API routes."""
        from .api import ChatRequest, ChatResponse, BuildResponse, StatusResponse
        
        # Determine auth dependency
        auth_dep = getattr(self, 'auth_dependency', None)
        dependencies = [Depends(auth_dep)] if auth_dep else []
        
        @self.app.post("/chat", response_model=ChatResponse, dependencies=dependencies)
        async def chat_endpoint(request: ChatRequest):
            """Chat with the RAG engine."""
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                raise HTTPException(status_code=400, detail="Pipeline not built. Call /build first.")
            
            try:
                result = self.pipeline.query(request.query)
                return ChatResponse(
                    query=request.query,
                    response=result,
                    session_id=request.session_id,
                    status="success"
                )
            except Exception as e:
                logging.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/build", response_model=BuildResponse, dependencies=dependencies)
        async def build_endpoint(background_tasks: BackgroundTasks):
            """Build the RAG pipeline."""
            try:
                from ..core.pipeline import Pipeline
                
                def build_pipeline():
                    self.pipeline = Pipeline(config=self.config)
                    self.pipeline.build()
                
                background_tasks.add_task(build_pipeline)
                
                return BuildResponse(
                    message="Pipeline build started",
                    status="building"
                )
            except Exception as e:
                logging.error(f"Build error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status", response_model=StatusResponse)
        async def status_endpoint():
            """Get pipeline status."""
            return StatusResponse(
                status="healthy",
                pipeline_built=hasattr(self, 'pipeline') and self.pipeline is not None,
                config=self.config.dict() if self.config else {}
            )
    
    def _setup_custom_routes(self):
        """Setup custom routes from configuration."""
        for route_config in self.api_config.custom_routes:
            path = route_config.get("path")
            methods = route_config.get("methods", ["GET"])
            handler = route_config.get("handler")
            
            if path and handler:
                for method in methods:
                    self.app.add_api_route(
                        path=path,
                        endpoint=handler,
                        methods=[method]
                    )
    
    def _setup_monitoring(self):
        """Setup monitoring endpoints."""
        if self.api_config.enable_metrics and self.metrics:
            @self.app.get(self.api_config.metrics_endpoint)
            async def metrics_endpoint():
                """Prometheus-style metrics endpoint."""
                return Response(
                    content=self.metrics.generate_prometheus_metrics(),
                    media_type="text/plain"
                )
        
        if self.api_config.enable_health_checks and self.health_checker:
            @self.app.get("/health")
            async def health_endpoint():
                """Health check endpoint."""
                health_status = self.health_checker.check_health()
                status_code = 200 if health_status["status"] == "healthy" else 503
                return JSONResponse(content=health_status, status_code=status_code)
            
            @self.app.get("/health/ready")
            async def readiness_endpoint():
                """Readiness check endpoint."""
                ready_status = self.health_checker.check_readiness()
                status_code = 200 if ready_status["ready"] else 503
                return JSONResponse(content=ready_status, status_code=status_code)
    
    def add_middleware(self, middleware_type: str, handler: Callable) -> None:
        """Add custom middleware to the application."""
        if middleware_type == "http":
            self.app.middleware("http")(handler)
        else:
            self.app.add_middleware(handler)
    
    def add_route(self, path: str, handler: Callable, methods: List[str], **kwargs) -> None:
        """Add a custom route to the application."""
        self.app.add_api_route(path=path, endpoint=handler, methods=methods, **kwargs)
    
    def set_error_handler(self, status_code: int, handler: Callable) -> None:
        """Set custom error handler."""
        self.app.exception_handler(status_code)(handler)
    
    def enable_authentication(self) -> None:
        """Enable authentication (already handled in _setup_authentication)."""
        pass
    
    def enable_rate_limiting(self) -> None:
        """Enable rate limiting (already handled in _setup_rate_limiting)."""
        pass
    
    def start_server(self, **kwargs) -> None:
        """Start the FastAPI server with production configuration."""
        if self.app is None:
            self.create_app()
        
        # Merge configuration
        server_config = {
            "app": self.app,
            "host": self.api_config.host,
            "port": self.api_config.port,
            "workers": self.api_config.workers,
            "reload": self.api_config.reload and self.api_config.debug,
            "log_level": "debug" if self.api_config.debug else "info",
        }
        server_config.update(kwargs)
        
        if server_config["workers"] > 1:
            # Multi-worker production mode
            uvicorn.run(
                "rag_engine.interfaces.fastapi_enhanced:create_production_app",
                **server_config
            )
        else:
            # Single worker development mode
            uvicorn.run(**server_config)


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
    
    server = FastAPIEnhanced(config=config, api_config=api_config)
    return server.create_app()
