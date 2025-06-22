"""
Django REST Framework implementation of the RAG Engine API server.
"""
import os
import sys
from typing import Optional
import django
from django.conf import settings
from django.urls import path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
import json
from .base_api import BaseAPIServer, APIModelFactory


class DjangoServer(BaseAPIServer):
    """Django REST Framework implementation of the RAG Engine API."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None
        self._configure_django()
    
    def _configure_django(self):
        """Configure Django settings programmatically."""
        if not settings.configured:
            settings.configure(
                DEBUG=True,
                SECRET_KEY='rag-engine-django-secret-key-change-in-production',
                INSTALLED_APPS=[
                    'django.contrib.contenttypes',
                    'django.contrib.auth',
                    'rest_framework',
                    'corsheaders',
                ],
                MIDDLEWARE=[
                    'corsheaders.middleware.CorsMiddleware',
                    'django.middleware.common.CommonMiddleware',
                    'django.middleware.csrf.CsrfViewMiddleware',
                ],
                ROOT_URLCONF=__name__,
                CORS_ALLOW_ALL_ORIGINS=True,  # Configure for production
                REST_FRAMEWORK={
                    'DEFAULT_RENDERER_CLASSES': [
                        'rest_framework.renderers.JSONRenderer',
                    ],
                    'DEFAULT_PARSER_CLASSES': [
                        'rest_framework.parsers.JSONParser',
                    ],
                }
            )
            django.setup()
    
    def create_app(self):
        """Create and configure the Django application."""
        # Django app is configured in _configure_django
        self.add_routes()
        return "django_app"  # Django doesn't return an app object like Flask/FastAPI
    
    def add_routes(self) -> None:
        """Add API routes to the Django application."""
        # Routes are defined via URL patterns below
        pass
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the Django server."""
        print(f"🚀 Starting Django server on http://{host}:{port}")
        print(f"📚 API Endpoints: http://{host}:{port}/")
        
        from django.core.management import execute_from_command_line
        
        # Set up Django command line arguments
        sys.argv = [
            'manage.py',
            'runserver',
            f'{host}:{port}',
        ]
        
        if kwargs.get("reload", False):
            sys.argv.append('--reload')
        
        try:
            execute_from_command_line(sys.argv)
        except Exception as e:
            print(f"❌ Failed to start Django server: {e}")


# Create a global instance for view functions
_server_instance = None

def set_server_instance(server):
    """Set the global server instance for view functions."""
    global _server_instance
    _server_instance = server


# Django view functions
@csrf_exempt
@require_http_methods(["GET"])
def root_view(request):
    """Root endpoint."""
    return JsonResponse({"message": "RAG Engine API (Django)", "version": "1.0.0"})


@csrf_exempt
@require_http_methods(["GET"])
def health_view(request):
    """Health check endpoint."""
    if _server_instance:
        return JsonResponse(_server_instance.handle_health())
    return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def status_view(request):
    """Get system status."""
    if _server_instance:
        return JsonResponse(_server_instance.handle_status())
    return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def build_view(request):
    """Build the RAG pipeline."""
    if not _server_instance:
        return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)
    
    result = _server_instance.handle_build()
    status_code = 500 if result["status"] == "error" else 200
    return JsonResponse(result, status=status_code)


@csrf_exempt
@require_http_methods(["POST"])
def chat_view(request):
    """Chat with the RAG system."""
    if not _server_instance:
        return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        if not data or "query" not in data:
            return JsonResponse({"error": "Missing 'query' in request body", "status": "error"}, status=400)
        
        query = data["query"]
        session_id = data.get("session_id")
        
        result = _server_instance.handle_chat(query, session_id)
        status_code = 500 if result["status"] == "error" else 200
        return JsonResponse(result, status=status_code)
    
    except Exception as e:
        return JsonResponse({"error": str(e), "status": "error"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def config_view(request):
    """Get current configuration (sanitized)."""
    if not _server_instance:
        return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)
    
    return JsonResponse({
        "documents": len(_server_instance.config.documents),
        "chunking_method": _server_instance.config.chunking.method,
        "embedding_provider": _server_instance.config.embedding.provider,
        "vectorstore_provider": _server_instance.config.vectorstore.provider,
        "llm_provider": _server_instance.config.llm.provider,
        "retrieval_top_k": _server_instance.config.retrieval.top_k
    })


@csrf_exempt
@require_http_methods(["GET"])
def documents_view(request):
    """List configured documents."""
    if not _server_instance:
        return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)
    
    if hasattr(_server_instance.pipeline, 'documents') and _server_instance.pipeline.documents:
        return JsonResponse({
            "documents": [
                {
                    "path": doc.get("path", "unknown"),
                    "type": doc.get("type", "unknown"),
                    "size": len(doc.get("content", ""))
                }
                for doc in _server_instance.pipeline.documents
            ],
            "total": len(_server_instance.pipeline.documents)
        })
    return JsonResponse({"documents": [], "total": 0})


@csrf_exempt
@require_http_methods(["GET"])
def chunks_view(request):
    """List document chunks."""
    if not _server_instance:
        return JsonResponse({"error": "Server not initialized", "status": "error"}, status=500)
    
    if hasattr(_server_instance.pipeline, 'chunks') and _server_instance.pipeline.chunks:
        return JsonResponse({
            "chunks": [
                {
                    "id": i,
                    "content_preview": chunk.get("content", "")[:100] + "..." if len(chunk.get("content", "")) > 100 else chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {})
                }
                for i, chunk in enumerate(_server_instance.pipeline.chunks)
            ],
            "total": len(_server_instance.pipeline.chunks)
        })
    return JsonResponse({"chunks": [], "total": 0})


# URL patterns for Django
urlpatterns = [
    path('', root_view, name='root'),
    path('health/', health_view, name='health'),
    path('status/', status_view, name='status'),
    path('build/', build_view, name='build'),
    path('chat/', chat_view, name='chat'),
    path('config/', config_view, name='config'),
    path('documents/', documents_view, name='documents'),
    path('chunks/', chunks_view, name='chunks'),
]


# Register Django server with the factory
try:
    APIModelFactory.register_server("django", DjangoServer)
except Exception:
    # Django not available
    pass
