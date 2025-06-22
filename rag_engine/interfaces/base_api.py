"""
Base API interface for supporting multiple web frameworks.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from rag_engine.core.orchestration import create_orchestrator, get_global_registry
from rag_engine.config.schema import RAGConfig
from rag_engine.config.loader import load_config


class BaseAPIServer(ABC):
    """Abstract base class for API servers supporting different frameworks."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[RAGConfig] = None, orchestrator_type: str = "default"):
        """Initialize the API server with configuration."""
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Initialize orchestrator (replaces hardcoded pipeline)
        self.orchestrator_type = orchestrator_type
        self.orchestrator = None
        self._is_built = False
        
        # Load component registry
        import rag_engine.core.component_registry  # Ensures components are registered
    
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
    def ensure_orchestrator_built(self) -> bool:
        """Ensure the orchestrator is built before handling requests."""
        if self.orchestrator is None:
            registry = get_global_registry()
            self.orchestrator = create_orchestrator(
                orchestrator_type=self.orchestrator_type,
                config=self.config,
                registry=registry
            )
            
        if not self._is_built:
            try:
                self.orchestrator.build()
                self._is_built = True
                return True
            except Exception as e:
                print(f"❌ Failed to build orchestrator: {str(e)}")
                return False
        return True
    
    def handle_chat(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle chat requests."""
        if not self.ensure_orchestrator_built():
            return {
                "error": "Orchestrator not ready",
                "answer": "Sorry, the system is not ready to answer questions.",
                "query": query,
                "status": "error"
            }
        
        try:
            response = self.orchestrator.query(query)
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
            if self.orchestrator is None:
                registry = get_global_registry()
                self.orchestrator = create_orchestrator(
                    orchestrator_type=self.orchestrator_type,
                    config=self.config,
                    registry=registry
                )
            
            self.orchestrator.build()
            self._is_built = True
            
            # Get component information for response
            documents_count = len(getattr(self.orchestrator, 'documents', []))
            chunks_count = len(getattr(self.orchestrator, 'chunks', []))
            
            return {
                "message": "Pipeline built successfully",
                "status": "success",
                "documents": documents_count,
                "chunks": chunks_count
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def handle_status(self) -> Dict[str, Any]:
        """Handle status requests."""
        if self.orchestrator:
            orchestrator_status = self.orchestrator.get_status()
            return {
                "status": "running",
                "pipeline_built": self._is_built,
                **orchestrator_status
            }
        else:
            return {
                "status": "running",
                "pipeline_built": self._is_built,
                "config": {
                    "embedding_provider": getattr(self.config.embedding, 'provider', 'default'),
                    "vectorstore_provider": getattr(self.config.vectorstore, 'provider', 'default'),
                    "llm_provider": getattr(self.config.llm, 'provider', 'default')
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
