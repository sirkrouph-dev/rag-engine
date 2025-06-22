"""
Base API interface for supporting multiple web frameworks.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.schema import RAGConfig
from rag_engine.config.loader import load_config


class BaseAPIServer(ABC):
    """Abstract base class for API servers supporting different frameworks."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[RAGConfig] = None):
        """Initialize the API server with configuration."""
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Don't initialize pipeline immediately - do it lazily
        self.pipeline = None
        self._is_built = False
    
    @abstractmethod
    def create_app(self) -> Any:
        """Create and configure the web application."""
        pass
    
    @abstractmethod
    def add_routes(self) -> None:
        """Add API routes to the application."""
        pass
    
    @abstractmethod
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the API server."""
        pass
    
    def ensure_pipeline_built(self) -> bool:
        """Ensure the pipeline is built before handling requests."""
        if self.pipeline is None:
            self.pipeline = Pipeline(self.config)
            
        if not self._is_built:
            try:
                self.pipeline.build()
                self._is_built = True
                return True
            except Exception as e:
                print(f"❌ Failed to build pipeline: {str(e)}")
                return False
        return True
    
    def handle_chat(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle chat requests."""
        if not self.ensure_pipeline_built():
            return {
                "error": "Pipeline not ready",
                "answer": "Sorry, the system is not ready to answer questions.",
                "query": query,
                "status": "error"
            }
        
        try:
            response = self.pipeline.query(query)
            return {
                "answer": response.get("answer", "No answer found"),
                "sources": response.get("sources", []),
                "query": query,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "Sorry, I encountered an error while processing your question.",
                "query": query,
                "status": "error"
            }
    
    def handle_build(self) -> Dict[str, Any]:
        """Handle pipeline build requests."""
        try:
            if self.pipeline is None:
                self.pipeline = Pipeline(self.config)
            
            self.pipeline.build()
            self._is_built = True
            return {
                "message": "Pipeline built successfully",
                "status": "success",
                "documents": len(self.pipeline.documents) if hasattr(self.pipeline, 'documents') else 0,
                "chunks": len(self.pipeline.chunks) if hasattr(self.pipeline, 'chunks') else 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def handle_status(self) -> Dict[str, Any]:
        """Handle status requests."""
        return {
            "status": "running",
            "pipeline_built": self._is_built,
            "config": {
                "embedding_provider": self.config.embedding.provider,
                "vectorstore_provider": self.config.vectorstore.provider,
                "llm_provider": self.config.llm.provider
            }
        }
    
    def handle_health(self) -> Dict[str, Any]:
        """Handle health check requests."""
        return {"status": "healthy", "timestamp": "2025-06-22T00:00:00Z"}


class APIModelFactory:
    """Factory for creating API server instances."""
    
    _servers = {}
    
    @classmethod
    def register_server(cls, framework: str, server_class: type):
        """Register a new API server framework."""
        cls._servers[framework] = server_class
    
    @classmethod
    def create_server(cls, framework: str, **kwargs) -> BaseAPIServer:
        """Create an API server instance for the specified framework."""
        if framework not in cls._servers:
            raise ValueError(f"Unknown framework: {framework}. Available: {list(cls._servers.keys())}")
        
        server_class = cls._servers[framework]
        return server_class(**kwargs)
    
    @classmethod
    def list_frameworks(cls) -> List[str]:
        """List all registered frameworks."""
        return list(cls._servers.keys())
