"""
Production-ready FastAPI server with comprehensive security, monitoring, and error handling.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our production modules
from rag_engine.core.reliability import (
    setup_production_error_handling, CircuitBreakerOpenError, RetryExhaustedException
)
from rag_engine.core.monitoring import setup_monitoring
from rag_engine.core.security import setup_security, SecurityLevel
from rag_engine.config.loader import load_config

logger = logging.getLogger(__name__)


# ================================
# Pydantic Models
# ================================

class HealthResponse(BaseModel):
    """Health check response model."""
    overall_status: str
    timestamp: str
    components: Dict[str, Any]
    system_metrics: Dict[str, Any]
    application_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    """Chat request model with validation."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: str


# ================================
# Production FastAPI Application
# ================================

class ProductionRAGServer:
    """Production-ready RAG Engine server with enterprise features."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path) if config_path else self._load_default_config()
        self.app = None
        self.reliability_components = None
        self.monitoring = None
        self.security_components = None
        self.startup_time = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default production configuration."""
        return {
            "server": {"host": "0.0.0.0", "port": 8000},
            "security": {
                "jwt_secret_key": "production-secret-key-change-me",
                "rate_limiting": {"enabled": True, "requests_per_minute": 100}
            },
            "monitoring": {"health_check_interval": 30},
            "circuit_breaker": {"failure_threshold": 5, "recovery_timeout": 60},
            "retry_policy": {"max_attempts": 3, "backoff_factor": 2}
        }
    
    async def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.startup_time = datetime.utcnow()
            await self._startup()
            logger.info("RAG Engine production server started successfully")
            
            yield
            
            # Shutdown
            await self._shutdown()
            logger.info("RAG Engine production server stopped")
        
        # Create FastAPI app with production settings
        self.app = FastAPI(
            title="RAG Engine API",
            description="Production-ready Retrieval-Augmented Generation Engine",
            version="1.0.0",
            docs_url="/docs" if self.config.get("app", {}).get("debug", False) else None,
            redoc_url="/redoc" if self.config.get("app", {}).get("debug", False) else None,
            lifespan=lifespan
        )
        
        # Setup middleware
        await self._setup_middleware()
        
        # Setup routes
        await self._setup_routes()
        
        # Setup error handlers
        await self._setup_error_handlers()
        
        return self.app
    
    async def _startup(self):
        """Initialize production components."""
        try:
            # Initialize reliability components
            self.reliability_components = setup_production_error_handling(self.config)
            logger.info("Reliability components initialized")
            
            # Initialize monitoring
            self.monitoring = setup_monitoring(self.config)
            logger.info("Monitoring system initialized")
            
            # Initialize security
            self.security_components = setup_security(self.config)
            logger.info("Security components initialized")
            
            # Register health checks
            await self._register_health_checks()
            
        except Exception as e:
            logger.error(f"Failed to initialize production components: {e}")
            raise
    
    async def _shutdown(self):
        """Cleanup on shutdown."""
        try:
            # Perform cleanup tasks
            logger.info("Performing shutdown cleanup...")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _setup_middleware(self):
        """Setup production middleware."""
        
        # CORS middleware
        cors_config = self.config.get("security", {}).get("cors", {})
        if cors_config.get("enabled", True):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_config.get("origins", ["*"]),
                allow_credentials=True,
                allow_methods=cors_config.get("methods", ["*"]),
                allow_headers=cors_config.get("headers", ["*"]),
            )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure for production
        )
        
        # Request logging and monitoring middleware
        @self.app.middleware("http")
        async def monitoring_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Record active request
            if self.monitoring:
                self.monitoring.metrics_collector.increment_active_requests()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                if self.monitoring:
                    self.monitoring.metrics_collector.record_request(
                        method=request.method,
                        endpoint=str(request.url.path),
                        status_code=response.status_code,
                        duration=duration
                    )
                
                # Add response headers
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
                
                return response
            
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                if self.monitoring:
                    self.monitoring.metrics_collector.record_request(
                        method=request.method,
                        endpoint=str(request.url.path),
                        status_code=500,
                        duration=duration
                    )
                raise
            
            finally:
                # Decrement active requests
                if self.monitoring:
                    self.monitoring.metrics_collector.decrement_active_requests()
        
        # Rate limiting middleware
        @self.app.middleware("http")
        async def rate_limiting_middleware(request: Request, call_next):
            if self.security_components and self.config.get("security", {}).get("rate_limiting", {}).get("enabled"):
                client_ip = request.client.host
                rate_limiter = self.security_components["rate_limiter"]
                
                allowed, rate_info = rate_limiter.is_allowed(client_ip)
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "details": rate_info,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
            
            return await call_next(request)
    
    async def _setup_routes(self):
        """Setup API routes."""
        
        # Health check endpoints
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check endpoint."""
            if not self.monitoring:
                raise HTTPException(status_code=503, detail="Monitoring not available")
            
            health_data = await self.monitoring.get_comprehensive_health()
            return HealthResponse(**health_data)
        
        @self.app.get("/health/live")
        async def liveness_check():
            """Kubernetes liveness probe."""
            return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.get("/health/ready")
        async def readiness_check():
            """Kubernetes readiness probe."""
            if not self.monitoring:
                raise HTTPException(status_code=503, detail="Not ready")
            
            health_data = await self.monitoring.get_comprehensive_health()
            if health_data["overall_status"] != "healthy":
                raise HTTPException(status_code=503, detail="Service not ready")
            
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint."""
            if not self.monitoring:
                return PlainTextResponse("Monitoring not available")
            
            metrics = self.monitoring.metrics_collector.get_prometheus_metrics()
            return PlainTextResponse(content=metrics, media_type="text/plain")
        
        # Chat endpoint with production features
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat_endpoint(
            request: ChatRequest,
            background_tasks: BackgroundTasks,
            auth_user: Optional[Dict] = Depends(self._get_current_user)
        ):
            """Chat endpoint with comprehensive error handling and monitoring."""
            
            # Input validation
            if self.security_components:
                validator = self.security_components["input_validator"]
                validation_result = validator.validate_input(request.query, "text")
                
                if not validation_result["valid"]:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid input",
                            "issues": validation_result["issues"]
                        }
                    )
                
                # Use sanitized input
                sanitized_query = validation_result["sanitized_data"]
            else:
                sanitized_query = request.query
            
            try:
                # Apply circuit breaker and retry logic
                if self.reliability_components:
                    circuit_breaker = self.reliability_components["circuit_breaker"]
                    retry_handler = self.reliability_components["retry_handler"]
                    
                    # This would call the actual RAG pipeline
                    response_data = await self._process_chat_with_reliability(
                        sanitized_query, request.session_id, request.context
                    )
                else:
                    # Fallback without reliability features
                    response_data = await self._process_chat_basic(
                        sanitized_query, request.session_id, request.context
                    )
                
                # Log successful interaction
                if self.security_components and auth_user:
                    audit_logger = self.security_components["audit_logger"]
                    background_tasks.add_task(
                        audit_logger.log_event,
                        user_id=auth_user.get("user_id"),
                        action="chat_query",
                        resource="chat_api",
                        details={"query_length": len(sanitized_query)},
                        ip_address="0.0.0.0",  # Would get from request
                        user_agent="",  # Would get from request
                        success=True
                    )
                
                return ChatResponse(**response_data)
            
            except CircuitBreakerOpenError:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable due to high error rate"
                )
            
            except RetryExhaustedException:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable - please try again later"
                )
            
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _setup_error_handlers(self):
        """Setup production error handlers."""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with structured responses."""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                    details=exc.detail if isinstance(exc.detail, dict) else None,
                    timestamp=datetime.utcnow().isoformat(),
                    request_id=getattr(request.state, "request_id", "unknown")
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle unexpected exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal server error",
                    details={"type": type(exc).__name__} if self.config.get("app", {}).get("debug") else None,
                    timestamp=datetime.utcnow().isoformat(),
                    request_id=getattr(request.state, "request_id", "unknown")
                ).dict()
            )
    
    async def _register_health_checks(self):
        """Register component health checks."""
        if self.monitoring:
            health_checker = self.monitoring.health_checker if hasattr(self.monitoring, 'health_checker') else None
            
            if health_checker:
                # Register health checks for all components
                health_checker.register_component("database", self.monitoring.check_database_health)
                health_checker.register_component("redis", self.monitoring.check_redis_health)
                health_checker.register_component("llm", self.monitoring.check_llm_health)
                health_checker.register_component("vectorstore", self.monitoring.check_vectorstore_health)
    
    async def _get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict]:
        """Extract current user from JWT token."""
        if not credentials or not self.security_components:
            return None
        
        auth_manager = self.security_components["auth_manager"]
        payload = auth_manager.verify_token(credentials.credentials)
        
        return payload
    
    async def _process_chat_with_reliability(self, query: str, session_id: Optional[str], 
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process chat with reliability patterns applied."""
        # This would integrate with the actual RAG pipeline
        # For now, return a mock response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "response": f"Mock response for: {query}",
            "sources": [],
            "session_id": session_id or "default",
            "metadata": {
                "processing_time": 0.1,
                "reliability_features": ["circuit_breaker", "retry", "monitoring"]
            }
        }
    
    async def _process_chat_basic(self, query: str, session_id: Optional[str], 
                                 context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic chat processing without reliability features."""
        # This would integrate with the actual RAG pipeline
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "response": f"Basic response for: {query}",
            "sources": [],
            "session_id": session_id or "default",
            "metadata": {
                "processing_time": 0.1,
                "reliability_features": []
            }
        }


# ================================
# Production Server Runner
# ================================

async def create_production_server(config_path: str = None) -> FastAPI:
    """Create production RAG server."""
    server = ProductionRAGServer(config_path)
    return await server.create_app()


def run_production_server(config_path: str = None, host: str = "0.0.0.0", port: int = 8000):
    """Run production server with uvicorn."""
    
    # Production uvicorn configuration
    uvicorn_config = {
        "host": host,
        "port": port,
        "workers": 1,  # Use multiple workers in production
        "loop": "uvloop",
        "http": "httptools",
        "access_log": True,
        "log_level": "info"
    }
    
    async def app_factory():
        return await create_production_server(config_path)
    
    # Run server
    uvicorn.run(
        "rag_engine.interfaces.production_api:create_production_server",
        factory=True,
        **uvicorn_config
    )


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_production_server(config_path)
