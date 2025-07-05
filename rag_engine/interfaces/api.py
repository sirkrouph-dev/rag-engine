"""
Production-ready FastAPI implementation with multi-worker support.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import subprocess
import tempfile
from .base_api import BaseAPIServer, APIModelFactory


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    query: str
    response: str
    session_id: Optional[str] = None
    status: str


class BuildResponse(BaseModel):
    """Response model for build endpoint."""
    message: str
    status: str
    documents: Optional[int] = None
    chunks: Optional[int] = None


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    pipeline_built: bool
    config: Dict[str, Any]


class AIAssistantRequest(BaseModel):
    """Request model for AI assistant endpoint."""
    question: str
    context: Optional[Dict[str, Any]] = None
    model: str = "phi3.5:latest"


class AIAssistantResponse(BaseModel):
    """Response model for AI assistant endpoint."""
    question: str
    response: str
    status: str
    context: Optional[Dict[str, Any]] = None


class StackConfigRequest(BaseModel):
    """Request model for stack configuration."""
    stack_type: str
    custom_requirements: Optional[List[str]] = None
    config_overrides: Optional[Dict[str, Any]] = None


class StackConfigResponse(BaseModel):
    """Response model for stack configuration."""
    stack_type: str
    requirements: List[str]
    estimated_size: str
    config: Dict[str, Any]
    status: str
    message: str


class StackAnalysisResponse(BaseModel):
    """Response model for stack analysis."""
    current_stack: str
    installed_packages: List[str]
    total_size: str
    recommendations: List[str]
    alternative_stacks: List[Dict[str, Any]]
    status: str


class DependencyAuditResponse(BaseModel):
    """Response model for dependency audit."""
    total_packages: int
    total_size: str
    unused_packages: List[str]
    heavy_packages: List[Dict[str, Any]]
    optimization_suggestions: List[str]
    status: str


class FastAPIServer(BaseAPIServer):
    """FastAPI implementation of the RAG Engine API with production scaling."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None
    
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="RAG Engine API",
            description="A modular Retrieval-Augmented Generation framework API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app = app
        self.add_routes()
        return app
    
    def add_routes(self) -> None:
        """Add API routes to the FastAPI application."""
        
        @self.app.get("/", tags=["Health"])
        async def root():
            """Root endpoint."""
            return {"message": "RAG Engine API", "version": "1.0.0"}
        
        @self.app.get("/health", tags=["Health"])
        async def health():
            """Health check endpoint."""
            return self.handle_health()
        
        @self.app.get("/status", response_model=StatusResponse, tags=["System"])
        async def status():
            """Get system status."""
            return self.handle_status()
        
        @self.app.post("/build", response_model=BuildResponse, tags=["Pipeline"])
        async def build_pipeline(background_tasks: BackgroundTasks):
            """Build the RAG pipeline."""
            result = self.handle_build()
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        
        @self.app.post("/chat", response_model=ChatResponse, tags=["Chat"])
        async def chat(request: ChatRequest):
            """Chat with the RAG system."""
            result = self.handle_chat(request.query, request.session_id)
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result

        @self.app.post("/ai-assistant", response_model=AIAssistantResponse, tags=["AI Assistant"])
        async def ai_assistant(request: AIAssistantRequest):
            """Ask the RAG Engine AI assistant for help and guidance."""
            try:
                # Import here to avoid dependency issues if ollama is not installed
                try:
                    import ollama
                except ImportError:
                    return AIAssistantResponse(
                        question=request.question,
                        response="AI assistant not available. Please install ollama-python: pip install ollama-python",
                        status="error"
                    )
                
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
                    {"role": "user", "content": request.question}
                ]
                
                # Add context if provided
                if request.context:
                    context_str = f"Current context: {json.dumps(request.context, indent=2)}"
                    conversation.insert(-1, {"role": "user", "content": context_str})
                
                # Get response from Ollama
                response = ollama.chat(
                    model=request.model,
                    messages=conversation
                )
                
                assistant_response = response['message']['content']
                
                return AIAssistantResponse(
                    question=request.question,
                    response=assistant_response,
                    status="success",
                    context=request.context
                )
                
            except Exception as e:
                return AIAssistantResponse(
                    question=request.question,
                    response=f"Error communicating with AI assistant: {str(e)}",
                    status="error"
                )

        @self.app.post("/stack/configure", response_model=StackConfigResponse, tags=["Stack Management"])
        async def configure_stack(request: StackConfigRequest):
            """Configure and install a specific stack."""
            try:
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
                
                if request.stack_type not in stacks:
                    return StackConfigResponse(
                        stack_type=request.stack_type,
                        requirements=[],
                        estimated_size="Unknown",
                        config={},
                        status="error",
                        message=f"Unknown stack type. Available: {', '.join(stacks.keys())}"
                    )
                
                stack_config = stacks[request.stack_type]
                requirements = stack_config["requirements"]
                
                # Add custom requirements if provided
                if request.custom_requirements:
                    requirements.extend(request.custom_requirements)
                
                # Create configuration
                config = {
                    "stack_type": request.stack_type,
                    "description": stack_config["description"],
                    "requirements_file": f"requirements-{request.stack_type.lower()}.txt"
                }
                
                # Apply config overrides
                if request.config_overrides:
                    config.update(request.config_overrides)
                
                # Write requirements file
                requirements_file = f"requirements-{request.stack_type.lower()}.txt"
                try:
                    with open(requirements_file, 'w') as f:
                        for req in requirements:
                            f.write(f"{req}\n")
                    
                    config["requirements_file_created"] = requirements_file
                    message = f"Stack {request.stack_type} configured successfully. Requirements saved to {requirements_file}"
                    
                except Exception as e:
                    message = f"Stack configured but failed to write requirements file: {str(e)}"
                
                return StackConfigResponse(
                    stack_type=request.stack_type,
                    requirements=requirements,
                    estimated_size=stack_config["estimated_size"],
                    config=config,
                    status="success",
                    message=message
                )
                
            except Exception as e:
                return StackConfigResponse(
                    stack_type=request.stack_type,
                    requirements=[],
                    estimated_size="Unknown",
                    config={},
                    status="error",
                    message=f"Error configuring stack: {str(e)}"
                )

        @self.app.get("/stack/analyze", response_model=StackAnalysisResponse, tags=["Stack Management"])
        async def analyze_stack():
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
                
                return StackAnalysisResponse(
                    current_stack=current_stack,
                    installed_packages=current_packages[:20],  # Limit for response size
                    total_size="Calculating...",  # Would need disk usage calculation
                    recommendations=recommendations,
                    alternative_stacks=alternative_stacks,
                    status="success"
                )
                
            except Exception as e:
                return StackAnalysisResponse(
                    current_stack="UNKNOWN",
                    installed_packages=[],
                    total_size="Unknown",
                    recommendations=[],
                    alternative_stacks=[],
                    status="error"
                )

        @self.app.get("/stack/audit", response_model=DependencyAuditResponse, tags=["Stack Management"])
        async def audit_dependencies():
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
                
                return DependencyAuditResponse(
                    total_packages=50,  # Would count actual packages
                    total_size="~1.2GB",  # Would calculate actual size
                    unused_packages=unused_packages,
                    heavy_packages=heavy_packages,
                    optimization_suggestions=optimization_suggestions,
                    status="success"
                )
                
            except Exception as e:
                return DependencyAuditResponse(
                    total_packages=0,
                    total_size="Unknown",
                    unused_packages=[],
                    heavy_packages=[],
                    optimization_suggestions=[],
                    status="error"
                )

        # Add orchestrator management endpoints
        @self.app.get("/orchestrator/status", tags=["Orchestrator"])
        async def orchestrator_status():
            """Get orchestrator status and component information."""
            if self.orchestrator:
                return self.orchestrator.get_status()
            return {"error": "Orchestrator not initialized"}
        
        @self.app.get("/orchestrator/components", tags=["Orchestrator"])
        async def list_components():
            """List available components."""
            from rag_engine.core.orchestration import get_global_registry
            registry = get_global_registry()
            return registry.list_components()
        
        @self.app.post("/orchestrator/rebuild", tags=["Orchestrator"])
        async def rebuild_orchestrator():
            """Rebuild the orchestrator with current configuration."""
            try:
                self._is_built = False
                self.orchestrator = None
                if self.ensure_orchestrator_built():
                    return {"message": "Orchestrator rebuilt successfully", "status": "success"}
                else:
                    return {"error": "Failed to rebuild orchestrator", "status": "error"}
            except Exception as e:
                return {"error": str(e), "status": "error"}
        
        @self.app.get("/orchestrator/components/{component_type}", tags=["Orchestrator"])
        async def get_component_status(component_type: str):
            """Get status of a specific component type."""
            if self.orchestrator and hasattr(self.orchestrator, 'components'):
                component = self.orchestrator.components.get(component_type)
                if component:
                    return {
                        "component_type": component_type,
                        "status": "active",
                        "class": component.__class__.__name__
                    }
                else:
                    return {
                        "component_type": component_type,
                        "status": "not_found"
                    }
            return {"error": "Orchestrator not initialized"}
        
        # AI Assistant endpoint
        @self.app.post("/ai/assist", response_model=AIAssistantResponse, tags=["AI"])
        async def ai_assist(request: AIAssistantRequest):
            """AI assistant query endpoint."""
            result = self.handle_ai_assistant(request.question, request.context, request.model)
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        
        # Stack configuration endpoints
        @self.app.post("/stack/configure", response_model=StackConfigResponse, tags=["Stack"])
        async def configure_stack(request: StackConfigRequest):
            """Configure a new stack."""
            result = self.handle_stack_configuration(request.stack_type, request.custom_requirements, request.config_overrides)
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        
        @self.app.get("/stack/analysis", response_model=StackAnalysisResponse, tags=["Stack"])
        async def analyze_stack():
            """Analyze current stack and provide recommendations."""
            result = self.handle_stack_analysis()
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        
        @self.app.get("/stack/audit", response_model=DependencyAuditResponse, tags=["Stack"])
        async def audit_dependencies():
            """Audit dependencies for the current stack."""
            result = self.handle_dependency_audit()
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])
            return result
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the FastAPI server with production-ready configuration."""
        if not self.app:
            self.create_app()
        
        # Production vs Development configuration
        workers = kwargs.get("workers", 1)
        reload = kwargs.get("reload", False)
        
        if workers > 1 and reload:
            print("⚠️  Warning: Auto-reload disabled when using multiple workers")
            reload = False
        
        print(f"🚀 Starting FastAPI server on http://{host}:{port}")
        print(f"👥 Workers: {workers}")
        print(f"📚 API Documentation: http://{host}:{port}/docs")
        print(f"📖 ReDoc Documentation: http://{host}:{port}/redoc")
        
        if workers > 1:
            # Production mode with multiple workers
            print(f"🏭 Production mode: {workers} workers for scalability")
            uvicorn.run(
                "rag_engine.interfaces.api_scalable:create_production_app",
                host=host,
                port=port,
                workers=workers,
                log_level=kwargs.get("log_level", "info"),
                access_log=kwargs.get("access_log", True)
            )
        else:
            # Development mode or single worker
            print("🛠️  Development mode: Single worker")
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level=kwargs.get("log_level", "info"),
                reload=reload,
                access_log=kwargs.get("access_log", True)
            )


def create_production_app():
    """Factory function to create a FastAPI app for production multi-worker deployment."""
    import os
    from ..config.loader import load_config
    
    # Load config from environment or default
    config_path = os.getenv("RAG_CONFIG_PATH", "config/production.json")
    
    try:
        config = load_config(config_path)
        server = FastAPIServer(config=config)
        return server.create_app()
    except Exception as e:
        # Fallback to minimal app if config loading fails
        print(f"⚠️  Config loading failed: {e}")
        print("🔧 Creating minimal FastAPI app")
        
        minimal_app = FastAPI(
            title="RAG Engine API (Minimal)",
            description="Production RAG Engine API - Configuration required",
            version="1.0.0"
        )
        
        @minimal_app.get("/health")
        def health():
            return {"status": "degraded", "message": "Configuration required"}
        
        @minimal_app.get("/")
        def root():
            return {"message": "RAG Engine API - Configuration required"}
        
        return minimal_app


# Register FastAPI server with the factory
APIModelFactory.register_server("fastapi", FastAPIServer)

# Backward compatibility
app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}
