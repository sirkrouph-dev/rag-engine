"""
Flask implementation of the RAG Engine API server.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional
import threading
from .base_api import BaseAPIServer, APIModelFactory


class FlaskServer(BaseAPIServer):
    """Flask implementation of the RAG Engine API."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None
    
    def create_app(self) -> Flask:
        """Create and configure the Flask application."""
        app = Flask(__name__)
        
        # Enable CORS
        CORS(app)
        
        # Configure JSON responses
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        
        self.app = app
        self.add_routes()
        return app
    
    def add_routes(self) -> None:
        """Add API routes to the Flask application."""
        
        @self.app.route("/", methods=["GET"])
        def root():
            """Root endpoint."""
            return jsonify({"message": "RAG Engine API (Flask)", "version": "1.0.0"})
        
        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify(self.handle_health())
        
        @self.app.route("/status", methods=["GET"])
        def status():
            """Get system status."""
            return jsonify(self.handle_status())
        
        @self.app.route("/build", methods=["POST"])
        def build_pipeline():
            """Build the RAG pipeline."""
            result = self.handle_build()
            status_code = 500 if result["status"] == "error" else 200
            return jsonify(result), status_code
        
        @self.app.route("/chat", methods=["POST"])
        def chat():
            """Chat with the RAG system."""
            try:
                data = request.get_json()
                if not data or "query" not in data:
                    return jsonify({"error": "Missing 'query' in request body", "status": "error"}), 400
                
                query = data["query"]
                session_id = data.get("session_id")
                
                result = self.handle_chat(query, session_id)
                status_code = 500 if result["status"] == "error" else 200
                return jsonify(result), status_code
            
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route("/config", methods=["GET"])
        def get_config():
            """Get current configuration (sanitized)."""
            return jsonify({
                "documents": len(self.config.documents),
                "chunking_method": self.config.chunking.method,
                "embedding_provider": self.config.embedding.provider,
                "vectorstore_provider": self.config.vectorstore.provider,
                "llm_provider": self.config.llm.provider,
                "retrieval_top_k": self.config.retrieval.top_k
            })
        
        @self.app.route("/documents", methods=["GET"])
        def list_documents():
            """List configured documents."""
            if hasattr(self.pipeline, 'documents') and self.pipeline.documents:
                return jsonify({
                    "documents": [
                        {
                            "path": doc.get("path", "unknown"),
                            "type": doc.get("type", "unknown"),
                            "size": len(doc.get("content", ""))
                        }
                        for doc in self.pipeline.documents
                    ],
                    "total": len(self.pipeline.documents)
                })
            return jsonify({"documents": [], "total": 0})
        
        @self.app.route("/chunks", methods=["GET"])
        def list_chunks():
            """List document chunks."""
            if hasattr(self.pipeline, 'chunks') and self.pipeline.chunks:
                return jsonify({
                    "chunks": [
                        {
                            "id": i,
                            "content_preview": chunk.get("content", "")[:100] + "..." if len(chunk.get("content", "")) > 100 else chunk.get("content", ""),
                            "metadata": chunk.get("metadata", {})
                        }
                        for i, chunk in enumerate(self.pipeline.chunks)
                    ],
                    "total": len(self.pipeline.chunks)
                })
            return jsonify({"chunks": [], "total": 0})
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Endpoint not found", "status": "error"}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({"error": "Internal server error", "status": "error"}), 500
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the Flask server with production-ready configuration."""
        if not self.app:
            self.create_app()
        
        workers = kwargs.get("workers", 1)
        debug = kwargs.get("debug", False)
        
        print(f"🚀 Starting Flask server on http://{host}:{port}")
        print(f"� Workers: {workers}")
        print(f"�📚 API Endpoints: http://{host}:{port}/")
        
        if workers > 1:
            # Production mode with Gunicorn
            print(f"🏭 Production mode: {workers} workers using Gunicorn")
            try:
                import subprocess
                import sys
                
                cmd = [
                    sys.executable, "-m", "gunicorn",
                    "--workers", str(workers),
                    "--worker-class", "gthread",
                    "--threads", "2",
                    "--bind", f"{host}:{port}",
                    "--access-logfile", "-",
                    "--error-logfile", "-",
                    "rag_engine.interfaces.flask_api:create_production_flask_app()"
                ]
                
                subprocess.run(cmd)
            except ImportError:
                print("⚠️  Gunicorn not available, falling back to development server")
                self.app.run(
                    host=host,
                    port=port,
                    debug=debug,
                    threaded=True
                )
        else:
            # Development mode
            print("🛠️  Development mode: Single threaded Flask server")
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=kwargs.get("threaded", True)
            )


def create_production_flask_app():
    """Factory function to create a Flask app for production deployment."""
    import os
    from ..config.loader import ConfigLoader
    
    # Load config from environment or default
    config_path = os.getenv("RAG_CONFIG_PATH", "config/production.json")
    
    try:
        config = ConfigLoader.load_config(config_path)
        server = FlaskServer(config=config)
        return server.create_app()
    except Exception as e:
        # Fallback to minimal app if config loading fails
        print(f"⚠️  Config loading failed: {e}")
        from flask import Flask
        
        minimal_app = Flask(__name__)
        
        @minimal_app.route("/health")
        def health():
            return {"status": "degraded", "message": "Configuration required"}
        
        @minimal_app.route("/")
        def root():
            return {"message": "RAG Engine Flask API - Configuration required"}
        
        return minimal_app


# Register Flask server with the factory
APIModelFactory.register_server("flask", FlaskServer)
