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
                 api_config: Optional[APICustomization] = None,
                 orchestrator_type: str = "default",
                 **kwargs):
        super().__init__(config_path, config, api_config)
        self.orchestrator_type = orchestrator_type
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
                    # Record the request with just the basic info
                    request_id = self.metrics.record_request(
                        endpoint=request.url.path,
                        method=request.method
                    )
                    # Record the response with status and timing
                    self.metrics.record_response(
                        request_id=request_id,
                        status_code=response.status_code,
                        response_time=process_time,
                        endpoint=request.url.path
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

        # Orchestrator endpoints
        @self.app.get("/orchestrator/status", dependencies=dependencies)
        async def orchestrator_status():
            """Get orchestrator status."""
            try:
                return {
                    "status": "active",
                    "type": "default",
                    "components_loaded": True,
                    "timestamp": time.time()
                }
            except Exception as e:
                logging.error(f"Orchestrator status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/orchestrator/components", dependencies=dependencies)
        async def get_components():
            """Get available components."""
            try:
                return {
                    "embedders": ["sentence-transformers", "openai", "vertex-ai"],
                    "vectorstores": ["faiss", "chroma", "pinecone"],
                    "llms": ["ollama", "openai", "anthropic"],
                    "chunkers": ["recursive", "semantic", "fixed-size"]
                }
            except Exception as e:
                logging.error(f"Components error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Document and chunk endpoints
        @self.app.get("/documents", dependencies=dependencies)
        async def get_documents():
            """Get document information."""
            try:
                # Mock response for now - would integrate with actual document store
                return {
                    "documents": [
                        {"id": "demo_doc_1", "name": "demo_document.md", "status": "processed"},
                        {"id": "test_doc_1", "name": "test_doc.txt", "status": "processed"}
                    ],
                    "total": 2,
                    "status": "success"
                }
            except Exception as e:
                logging.error(f"Documents error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/chunks", dependencies=dependencies)
        async def get_chunks():
            """Get chunk information."""
            try:
                # Mock response for now - would integrate with actual vector store
                return {
                    "chunks": [
                        {"id": "chunk_1", "document_id": "demo_doc_1", "size": 512},
                        {"id": "chunk_2", "document_id": "demo_doc_1", "size": 485},
                        {"id": "chunk_3", "document_id": "test_doc_1", "size": 298}
                    ],
                    "total": 3,
                    "status": "success"
                }
            except Exception as e:
                logging.error(f"Chunks error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # AI Assistant and Stack Management routes
        @self.app.post("/ai-assistant")
        async def ai_assistant_endpoint(request: dict):
            """Ask the RAG Engine AI assistant for help and guidance."""
            try:
                # Import here to avoid dependency issues if ollama is not installed
                try:
                    import ollama
                except ImportError:
                    return {
                        "question": request.get("question", ""),
                        "response": "AI assistant not available. Please install ollama-python: pip install ollama-python",
                        "status": "error"
                    }
                
                question = request.get("question", "")
                context = request.get("context")
                model = request.get("model", "phi3.5:latest")
                
                # System prompt for the assistant
                system_prompt = """You are a helpful RAG Engine assistant specialized in package bloat management and optimization. You provide support for users of the RAG Engine framework.

STACK INFORMATION:
- DEMO: Quick demos (~200MB) - Minimal deps: ollama-python, sentence-transformers, faiss-cpu, typer, rich
- LOCAL: Local development (~500MB) - Adds: transformers, torch, multiple vector stores, advanced chunking  
- CLOUD: Production APIs (~100MB) - Cloud APIs only: openai, anthropic, requests, minimal local processing
- MINI: Embedded systems (~50MB) - Ultra minimal: only core logic, no UI, basic text processing
- FULL: Everything (~1GB) - All features: research models, advanced chunking, multiple frameworks
- RESEARCH: Academic (~1GB+) - Cutting-edge: experimental models, specialized libraries

BLOAT MANAGEMENT STRATEGIES:
1. **Tiered Requirements**: Use requirements-{stack}.txt files for different use cases
2. **Optional Dependencies**: Install only what's needed with pip extras: pip install rag-engine[demo]
3. **Lazy Imports**: Import heavy libraries only when needed at runtime  
4. **Runtime Detection**: Auto-detect available packages and gracefully fallback
5. **Plugin Architecture**: Load components dynamically based on user needs
6. **Dependency Analysis**: Help users understand what each package does and if they need it

PACKAGE OPTIMIZATION:
- Suggest lighter alternatives (e.g., sentence-transformers vs full transformers)
- Identify unused dependencies in current setup
- Recommend stack switches for user needs
- Help with dependency conflicts and version pinning
- Guide on Docker vs local installs for bloat reduction

You can help with:
- Configuration questions and stack recommendations
- Package bloat analysis and optimization suggestions
- Installation and setup guidance
- Performance optimization
- Troubleshooting common issues

Keep responses concise and actionable. Focus on practical solutions."""

                # Prepare conversation context
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Add context if provided
                if context:
                    import json
                    context_str = f"Current context: {json.dumps(context, indent=2)}"
                    conversation.insert(-1, {"role": "user", "content": context_str})
                
                # Get response from Ollama
                response = ollama.chat(
                    model=model,
                    messages=conversation
                )
                
                assistant_response = response['message']['content']
                
                return {
                    "question": question,
                    "response": assistant_response,
                    "status": "success",
                    "context": context
                }
                
            except Exception as e:
                return {
                    "question": request.get("question", ""),
                    "response": f"Error communicating with AI assistant: {str(e)}",
                    "status": "error"
                }

        @self.app.post("/stack/configure")
        async def configure_stack_endpoint(request: dict):
            """Configure and install a specific stack."""
            try:
                stack_type = request.get("stack_type")
                custom_requirements = request.get("custom_requirements", [])
                config_overrides = request.get("config_overrides", {})
                
                # Stack definitions
                stacks = {
                    "DEMO": {
                        "requirements": [
                            "ollama-python>=0.2.0",
                            "sentence-transformers>=2.2.0",
                            "faiss-cpu>=1.7.0",
                            "typer>=0.9.0",
                            "rich>=13.0.0",
                            "pydantic>=2.0.0",
                            "fastapi>=0.104.0",
                            "uvicorn>=0.24.0"
                        ],
                        "estimated_size": "~200MB",
                        "description": "Quick demos with minimal dependencies"
                    },
                    "LOCAL": {
                        "requirements": [
                            "transformers>=4.30.0",
                            "torch>=2.0.0",
                            "sentence-transformers>=2.2.0",
                            "faiss-cpu>=1.7.0",
                            "chromadb>=0.4.0",
                            "langchain>=0.0.300",
                            "ollama-python>=0.2.0",
                            "typer>=0.9.0",
                            "rich>=13.0.0",
                            "pydantic>=2.0.0",
                            "fastapi>=0.104.0",
                            "uvicorn>=0.24.0"
                        ],
                        "estimated_size": "~500MB",
                        "description": "Local development with multiple vector stores"
                    },
                    "CLOUD": {
                        "requirements": [
                            "openai>=1.0.0",
                            "anthropic>=0.7.0",
                            "requests>=2.31.0",
                            "typer>=0.9.0",
                            "pydantic>=2.0.0",
                            "fastapi>=0.104.0",
                            "uvicorn>=0.24.0"
                        ],
                        "estimated_size": "~100MB",
                        "description": "Cloud APIs only, minimal local processing"
                    },
                    "MINI": {
                        "requirements": [
                            "pydantic>=2.0.0",
                            "typer>=0.9.0"
                        ],
                        "estimated_size": "~50MB",
                        "description": "Ultra minimal for embedded systems"
                    },
                    "FULL": {
                        "requirements": [
                            "transformers>=4.30.0",
                            "torch>=2.0.0",
                            "sentence-transformers>=2.2.0",
                            "faiss-cpu>=1.7.0",
                            "chromadb>=0.4.0",
                            "langchain>=0.0.300",
                            "openai>=1.0.0",
                            "anthropic>=0.7.0",
                            "ollama-python>=0.2.0",
                            "pinecone-client>=2.2.0",
                            "weaviate-client>=3.25.0",
                            "pymongo>=4.5.0",
                            "redis>=5.0.0",
                            "typer>=0.9.0",
                            "rich>=13.0.0",
                            "pydantic>=2.0.0",
                            "fastapi>=0.104.0",
                            "uvicorn>=0.24.0"
                        ],
                        "estimated_size": "~1GB",
                        "description": "All features for production use"
                    },
                    "RESEARCH": {
                        "requirements": [
                            "transformers>=4.30.0",
                            "torch>=2.0.0",
                            "sentence-transformers>=2.2.0",
                            "faiss-cpu>=1.7.0",
                            "chromadb>=0.4.0",
                            "langchain>=0.0.300",
                            "openai>=1.0.0",
                            "anthropic>=0.7.0",
                            "ollama-python>=0.2.0",
                            "datasets>=2.14.0",
                            "evaluate>=0.4.0",
                            "accelerate>=0.23.0",
                            "wandb>=0.15.0",
                            "tensorboard>=2.14.0",
                            "typer>=0.9.0",
                            "rich>=13.0.0",
                            "pydantic>=2.0.0",
                            "fastapi>=0.104.0",
                            "uvicorn>=0.24.0"
                        ],
                        "estimated_size": "~1GB+",
                        "description": "Cutting-edge research with experimental models"
                    }
                }
                
                if stack_type not in stacks:
                    return {
                        "stack_type": stack_type,
                        "requirements": [],
                        "estimated_size": "Unknown",
                        "config": {},
                        "status": "error",
                        "message": f"Unknown stack type. Available: {', '.join(stacks.keys())}"
                    }
                
                stack_config = stacks[stack_type]
                requirements = stack_config["requirements"]
                
                # Add custom requirements if provided
                if custom_requirements:
                    requirements.extend(custom_requirements)
                
                # Create configuration
                config = {
                    "stack_type": stack_type,
                    "description": stack_config["description"],
                    "requirements_file": f"requirements-{stack_type.lower()}.txt"
                }
                
                # Apply config overrides
                if config_overrides:
                    config.update(config_overrides)
                
                # Write requirements file
                requirements_file = f"requirements-{stack_type.lower()}.txt"
                try:
                    with open(requirements_file, 'w') as f:
                        for req in requirements:
                            f.write(f"{req}\n")
                    
                    config["requirements_file_created"] = requirements_file
                    message = f"Stack {stack_type} configured successfully. Requirements saved to {requirements_file}"
                    
                except Exception as e:
                    message = f"Stack configured but failed to write requirements file: {str(e)}"
                
                return {
                    "stack_type": stack_type,
                    "requirements": requirements,
                    "estimated_size": stack_config["estimated_size"],
                    "config": config,
                    "status": "success",
                    "message": message
                }
                
            except Exception as e:
                return {
                    "stack_type": request.get("stack_type", ""),
                    "requirements": [],
                    "estimated_size": "Unknown",
                    "config": {},
                    "status": "error",
                    "message": f"Error configuring stack: {str(e)}"
                }

        @self.app.get("/stack/analyze")
        async def analyze_stack_endpoint():
            """Analyze current stack and provide recommendations."""
            try:
                # Try to determine current stack by checking installed packages
                current_packages = []
                try:
                    import subprocess
                    result = subprocess.run(
                        ["pip", "list", "--format=json"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    import json
                    packages_info = json.loads(result.stdout)
                    current_packages = [pkg["name"] for pkg in packages_info]
                except Exception:
                    current_packages = []
                
                # Analyze which stack type this resembles
                stack_indicators = {
                    "DEMO": ["ollama-python", "sentence-transformers", "faiss-cpu"],
                    "LOCAL": ["transformers", "torch", "chromadb"],
                    "CLOUD": ["openai", "anthropic"],
                    "MINI": [],  # Minimal packages
                    "FULL": ["pinecone-client", "weaviate-client", "redis"],
                    "RESEARCH": ["datasets", "evaluate", "wandb", "tensorboard"]
                }
                
                current_stack = "UNKNOWN"
                max_matches = 0
                
                for stack, indicators in stack_indicators.items():
                    matches = sum(1 for indicator in indicators if indicator in current_packages)
                    if matches > max_matches:
                        max_matches = matches
                        current_stack = stack
                
                # Generate recommendations
                recommendations = []
                alternative_stacks = []
                
                if "torch" in current_packages and "transformers" not in current_packages:
                    recommendations.append("Consider switching to sentence-transformers for lighter weight")
                
                if "faiss-gpu" in current_packages:
                    recommendations.append("Using faiss-gpu - consider faiss-cpu for smaller footprint")
                
                if len(current_packages) > 100:
                    recommendations.append("Large number of packages detected - consider stack optimization")
                
                # Suggest alternative stacks
                if current_stack == "FULL":
                    alternative_stacks.append({
                        "name": "LOCAL", 
                        "savings": "~500MB",
                        "reason": "Remove cloud dependencies if not needed"
                    })
                elif current_stack == "LOCAL":
                    alternative_stacks.append({
                        "name": "DEMO", 
                        "savings": "~300MB",
                        "reason": "Use for quick demos and testing"
                    })
                
                return {
                    "current_stack": current_stack,
                    "installed_packages": current_packages[:20],  # Limit for response size
                    "total_size": "Calculating...",  # Would need disk usage calculation
                    "recommendations": recommendations,
                    "alternative_stacks": alternative_stacks,
                    "status": "success"
                }
                
            except Exception as e:
                return {
                    "current_stack": "UNKNOWN",
                    "installed_packages": [],
                    "total_size": "Unknown",
                    "recommendations": [],
                    "alternative_stacks": [],
                    "status": "error"
                }

        @self.app.get("/stack/audit")
        async def audit_dependencies_endpoint():
            """Audit dependencies and identify optimization opportunities."""
            try:
                # Get installed packages with sizes
                heavy_packages = []
                unused_packages = []
                optimization_suggestions = []
                
                # This would typically use pipdeptree or similar for real analysis
                # For now, provide example suggestions
                optimization_suggestions = [
                    "Consider using sentence-transformers instead of full transformers for embeddings",
                    "Use faiss-cpu instead of faiss-gpu if GPU acceleration isn't needed",
                    "Remove development dependencies in production",
                    "Use slim Docker base images",
                    "Enable lazy imports for heavy libraries"
                ]
                
                # Example heavy packages (would be calculated from actual environment)
                heavy_packages = [
                    {"name": "torch", "size": "500MB", "reason": "Deep learning framework"},
                    {"name": "transformers", "size": "200MB", "reason": "Model library"},
                    {"name": "tensorflow", "size": "400MB", "reason": "Alternative to torch"}
                ]
                
                return {
                    "total_packages": 50,  # Would count actual packages
                    "total_size": "~1.2GB",  # Would calculate actual size
                    "unused_packages": unused_packages,
                    "heavy_packages": heavy_packages,
                    "optimization_suggestions": optimization_suggestions,
                    "status": "success"
                }
                
            except Exception as e:
                return {
                    "total_packages": 0,
                    "total_size": "Unknown",
                    "unused_packages": [],
                    "heavy_packages": [],
                    "optimization_suggestions": [],
                    "status": "error"
                }
    
    def _setup_custom_routes(self):
        """Setup custom routes from configuration."""
        # Setup conversational routing endpoints
        self._setup_routing_endpoints()
        
        # Setup custom routes from configuration
        if not self.api_config.custom_routes:
            return
            
        for route_config in self.api_config.custom_routes:
            path = route_config.get("path")
            methods = route_config.get("methods", ["GET"])
            handler = route_config.get("handler")
            
            if path and handler:
                for method in methods:
                    self.app.add_api_route(
                        path=path,
                        endpoint=handler,
                        methods=[method],
                        **route_config.get("kwargs", {})
                    )

    def _setup_routing_endpoints(self):
        """Setup conversational routing and template management endpoints."""
        # Determine auth dependency
        auth_dep = getattr(self, 'auth_dependency', None)
        dependencies = [Depends(auth_dep)] if auth_dep else []

        @self.app.get("/routing/templates", dependencies=dependencies)
        async def get_routing_templates():
            """Get all available routing templates."""
            try:
                from pathlib import Path
                import os
                
                templates_dir = Path("templates/routing")
                if not templates_dir.exists():
                    return {"templates": [], "message": "No routing templates found"}
                
                templates = {}
                for template_file in templates_dir.glob("*.txt"):
                    try:
                        with open(template_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        templates[template_file.stem] = {
                            "name": template_file.stem,
                            "filename": template_file.name,
                            "content": content,
                            "path": str(template_file)
                        }
                    except Exception as e:
                        logging.error(f"Error reading template {template_file}: {e}")
                
                return {"templates": templates}
            except Exception as e:
                logging.error(f"Error getting routing templates: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/routing/templates/{template_name}", dependencies=dependencies)
        async def get_routing_template(template_name: str):
            """Get a specific routing template."""
            try:
                from pathlib import Path
                
                template_path = Path(f"templates/routing/{template_name}.txt")
                if not template_path.exists():
                    raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
                
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return {
                    "name": template_name,
                    "filename": template_path.name,
                    "content": content,
                    "path": str(template_path)
                }
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Error getting template {template_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/routing/templates/{template_name}", dependencies=dependencies)
        async def update_routing_template(template_name: str, template_data: dict):
            """Update a routing template."""
            try:
                from pathlib import Path
                import os
                
                # Validate input
                if 'content' not in template_data:
                    raise HTTPException(status_code=400, detail="Template content is required")
                
                # Ensure templates directory exists
                templates_dir = Path("templates/routing")
                templates_dir.mkdir(parents=True, exist_ok=True)
                
                template_path = templates_dir / f"{template_name}.txt"
                
                # Backup existing template if it exists
                if template_path.exists():
                    backup_path = templates_dir / f"{template_name}.txt.backup"
                    import shutil
                    shutil.copy2(template_path, backup_path)
                
                # Write new content
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_data['content'])
                
                return {
                    "message": f"Template '{template_name}' updated successfully",
                    "path": str(template_path),
                    "backup_created": template_path.exists()
                }
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Error updating template {template_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/routing/config", dependencies=dependencies)
        async def get_routing_config():
            """Get current conversational routing configuration."""
            try:
                # Get routing config from current configuration
                routing_config = {}
                if hasattr(self, 'config') and self.config:
                    config_dict = self.config.dict() if hasattr(self.config, 'dict') else self.config
                    prompting_config = config_dict.get('prompting', {})
                    
                    if prompting_config.get('type') == 'conversational_rag':
                        routing_config = {
                            "enabled": prompting_config.get('enable_routing', True),
                            "fallback_to_simple": prompting_config.get('fallback_to_simple', True),
                            "routing_config": prompting_config.get('routing_config', {}),
                            "domain_config": prompting_config.get('domain_config', {}),
                            "system_prompt": prompting_config.get('system_prompt', '')
                        }
                    else:
                        routing_config = {
                            "enabled": False,
                            "message": "Conversational routing not currently configured"
                        }
                
                return {"routing_config": routing_config}
            except Exception as e:
                logging.error(f"Error getting routing config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/routing/config", dependencies=dependencies)
        async def update_routing_config(config_data: dict):
            """Update conversational routing configuration."""
            try:
                # This would typically update the configuration and reload the prompter
                # For now, we'll return a placeholder response
                return {
                    "message": "Routing configuration updated",
                    "note": "Configuration updates require server restart to take full effect",
                    "config": config_data
                }
            except Exception as e:
                logging.error(f"Error updating routing config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/routing/test", dependencies=dependencies)
        async def test_routing(test_data: dict):
            """Test routing with a sample query."""
            try:
                query = test_data.get('query', '')
                if not query:
                    raise HTTPException(status_code=400, detail="Query is required for testing")
                
                # Import conversational routing components
                from ..core.conversational_routing import ConversationalRouter
                from ..core.conversational_integration import ConversationalRAGPrompter
                
                # Create a test router with current config
                routing_config = test_data.get('config', {})
                test_router = ConversationalRouter(routing_config)
                
                # Get routing insights without actually processing
                insights = {
                    "query": query,
                    "routing_enabled": True,
                    "estimated_category": "rag_factual",  # Placeholder
                    "estimated_strategy": "rag_retrieval",  # Placeholder
                    "confidence": 0.85,
                    "reasoning": "Test routing analysis - would require LLM for actual analysis"
                }
                
                return {"insights": insights}
            except HTTPException:
                raise  # Re-raise HTTP exceptions as-is
            except Exception as e:
                logging.error(f"Error testing routing: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/routing/analytics", dependencies=dependencies)
        async def get_routing_analytics():
            """Get routing analytics and usage statistics."""
            try:
                # Placeholder analytics - would be populated from actual usage tracking
                analytics = {
                    "total_queries": 0,
                    "routing_decisions": {
                        "rag_retrieval": 0,
                        "contextual_chat": 0,
                        "simple_response": 0,
                        "polite_rejection": 0,
                        "clarification_request": 0
                    },
                    "category_distribution": {
                        "rag_factual": 0,
                        "greeting": 0,
                        "out_of_context": 0,
                        "follow_up": 0
                    },
                    "avg_confidence": 0.0,
                    "template_usage": {}
                }
                
                return {"analytics": analytics}
            except Exception as e:
                logging.error(f"Error getting routing analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_monitoring(self):
        """Setup monitoring endpoints."""
        if self.api_config.enable_metrics and self.metrics:
            @self.app.get(self.api_config.metrics_endpoint)
            async def metrics_endpoint():
                """Get system metrics."""
                return self.metrics.get_metrics()
        
        if self.api_config.enable_health_checks:
            @self.app.get("/health")
            async def health_endpoint():
                """Health check endpoint."""
                return {"status": "healthy", "timestamp": time.time()}

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


