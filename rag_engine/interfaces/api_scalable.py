"""
Production-ready FastAPI implementation with multi-worker support.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
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
        
        @self.app.get("/config", tags=["System"])
        async def get_config():
            """Get current configuration (sanitized)."""
            return {
                "documents": len(self.config.documents),
                "chunking_method": self.config.chunking.method,
                "embedding_provider": self.config.embedding.provider,
                "vectorstore_provider": self.config.vectorstore.provider,
                "llm_provider": self.config.llm.provider,
                "retrieval_top_k": self.config.retrieval.top_k
            }
        
        @self.app.get("/documents", tags=["Documents"])
        async def list_documents():
            """List configured documents."""
            if hasattr(self.pipeline, 'documents') and self.pipeline.documents:
                return {
                    "documents": [
                        {
                            "path": doc.get("path", "unknown"),
                            "type": doc.get("type", "unknown"),
                            "size": len(doc.get("content", ""))
                        }
                        for doc in self.pipeline.documents
                    ],
                    "total": len(self.pipeline.documents)
                }
            return {"documents": [], "total": 0}
        
        @self.app.get("/chunks", tags=["Documents"])
        async def list_chunks():
            """List document chunks."""
            if hasattr(self.pipeline, 'chunks') and self.pipeline.chunks:
                return {
                    "chunks": [
                        {
                            "id": i,
                            "content_preview": chunk.get("content", "")[:100] + "..." if len(chunk.get("content", "")) > 100 else chunk.get("content", ""),
                            "metadata": chunk.get("metadata", {})
                        }
                        for i, chunk in enumerate(self.pipeline.chunks)
                    ],
                    "total": len(self.pipeline.chunks)
                }
            return {"chunks": [], "total": 0}
    
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
