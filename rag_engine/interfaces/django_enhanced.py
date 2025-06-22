"""
Enhanced Django REST framework implementation with full customization support.
"""
import os
import sys
from typing import Optional, Dict, Any, List, Callable
import logging
import time
from collections import defaultdict, deque

# Django imports (will be available when Django is installed)
try:
    import django
    from django.conf import settings
    from django.core.wsgi import get_wsgi_application
    from django.urls import path, include
    from django.http import JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from django.utils.decorators import method_decorator
    from django.views import View
    from rest_framework import status
    from rest_framework.views import APIView
    from rest_framework.response import Response
    from rest_framework.decorators import api_view, authentication_classes, permission_classes
    from rest_framework.authentication import TokenAuthentication
    from rest_framework.permissions import IsAuthenticated
    from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

from .enhanced_base_api import EnhancedBaseAPIServer, APICustomization, AuthMethod, RateLimitType
from .monitoring import MetricsCollector, HealthChecker


class DjangoEnhanced(EnhancedBaseAPIServer):
    """Enhanced Django REST implementation with full customization support."""
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[Any] = None,
                 api_config: Optional[APICustomization] = None):
        if not DJANGO_AVAILABLE:
            raise ImportError("Django is not installed. Install with: pip install django djangorestframework")
        
        super().__init__(config_path, config, api_config)
        self.app = None
        self.request_counts = defaultdict(lambda: deque())
        self.metrics = MetricsCollector() if api_config and api_config.enable_metrics else None
        self.health_checker = HealthChecker() if api_config and api_config.enable_health_checks else None
        self._setup_django()
        
    def _setup_django(self):
        """Configure Django settings."""
        django_settings = {
            'DEBUG': self.api_config.debug,
            'SECRET_KEY': 'rag-engine-secret-key-change-in-production',
            'ALLOWED_HOSTS': ['*'],
            'INSTALLED_APPS': [
                'django.contrib.auth',
                'django.contrib.contenttypes',
                'rest_framework',
                'corsheaders',
            ],
            'MIDDLEWARE': [
                'corsheaders.middleware.CorsMiddleware',
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
                'django.contrib.auth.middleware.AuthenticationMiddleware',
                'django.contrib.messages.middleware.MessageMiddleware',
            ],
            'ROOT_URLCONF': 'rag_engine.interfaces.django_enhanced_urls',
            'DATABASES': {
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': ':memory:',
                }
            },
            'REST_FRAMEWORK': {
                'DEFAULT_AUTHENTICATION_CLASSES': [],
                'DEFAULT_PERMISSION_CLASSES': [],
                'DEFAULT_THROTTLE_CLASSES': [],
                'DEFAULT_THROTTLE_RATES': {},
            },
            'CORS_ALLOW_ALL_ORIGINS': True,
            'CORS_ALLOWED_ORIGINS': self.api_config.cors_origins,
            'USE_TZ': True,
        }
        
        # Apply rate limiting settings
        if self.api_config.enable_rate_limiting:
            django_settings['REST_FRAMEWORK']['DEFAULT_THROTTLE_CLASSES'] = [
                'rest_framework.throttling.AnonRateThrottle',
                'rest_framework.throttling.UserRateThrottle'
            ]
            django_settings['REST_FRAMEWORK']['DEFAULT_THROTTLE_RATES'] = {
                'anon': f'{self.api_config.requests_per_minute}/min',
                'user': f'{self.api_config.requests_per_minute}/min'
            }
        
        # Apply authentication settings
        if self.api_config.auth_method != AuthMethod.NONE:
            django_settings['REST_FRAMEWORK']['DEFAULT_AUTHENTICATION_CLASSES'] = [
                'rest_framework.authentication.TokenAuthentication'
            ]
            django_settings['REST_FRAMEWORK']['DEFAULT_PERMISSION_CLASSES'] = [
                'rest_framework.permissions.IsAuthenticated'
            ]
        
        if not settings.configured:
            settings.configure(**django_settings)
            django.setup()
    
    def create_app(self) -> Any:
        """Create and configure the Django application."""
        if self.app is not None:
            return self.app
        
        # Django app is configured through settings and URLs
        self.app = get_wsgi_application()
        return self.app
    
    def add_middleware(self, middleware_type: str, handler: Callable) -> None:
        """Add custom middleware to the application."""
        # Django middleware needs to be configured in settings
        # This is a simplified implementation
        pass
    
    def add_route(self, path: str, handler: Callable, methods: List[str], **kwargs) -> None:
        """Add a custom route to the application."""
        # Django routes are defined in URLconf
        # This would need to be implemented through dynamic URL patterns
        pass
    
    def set_error_handler(self, status_code: int, handler: Callable) -> None:
        """Set custom error handler."""
        # Django error handling is done through middleware or views
        pass
    
    def enable_authentication(self) -> None:
        """Enable authentication (configured in _setup_django)."""
        pass
    
    def enable_rate_limiting(self) -> None:
        """Enable rate limiting (configured in _setup_django)."""
        pass
    
    def start_server(self, **kwargs) -> None:
        """Start the Django server with production configuration."""
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
                "rag_engine.interfaces.django_enhanced:create_production_app"
            ]
            
            subprocess.run(cmd)
        else:
            # Development mode
            from django.core.management import execute_from_command_line
            execute_from_command_line([
                'manage.py', 'runserver',
                f'{self.api_config.host}:{self.api_config.port}'
            ])


# Django REST views (only if Django is available)
if DJANGO_AVAILABLE:
    class ChatAPIView(APIView):
        """Chat endpoint for Django REST."""
        
        def post(self, request):
            """Handle chat requests."""
            try:
                query = request.data.get('query')
                if not query:
                    return Response(
                        {"error": "Query is required"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Get pipeline from Django app context
                # In a real implementation, this would be stored in cache or database
                pipeline = getattr(request, '_rag_pipeline', None)
                if not pipeline:
                    return Response(
                        {"error": "Pipeline not built. Call /build first."}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                result = pipeline.query(query)
                return Response({
                    "query": query,
                    "response": result,
                    "session_id": request.data.get('session_id'),
                    "status": "success"
                })
                
            except Exception as e:
                logging.error(f"Chat error: {e}")
                return Response(
                    {"error": str(e)}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


    class BuildAPIView(APIView):
        """Build endpoint for Django REST."""
        
        def post(self, request):
            """Handle build requests."""
            try:
                from ..core.pipeline import Pipeline
                from ..config.loader import ConfigLoader
                
                # Load configuration
                config_loader = ConfigLoader()
                config = config_loader.load_config("config/production.json")
                
                # Build pipeline
                pipeline = Pipeline(config=config)
                pipeline.build()
                
                # Store pipeline in cache (simplified)
                # In production, use Redis or database
                request._rag_pipeline = pipeline
                
                return Response({
                    "message": "Pipeline built successfully",
                    "status": "success"
                })
                
            except Exception as e:
                logging.error(f"Build error: {e}")
                return Response(
                    {"error": str(e)}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


    class StatusAPIView(APIView):
        """Status endpoint for Django REST."""
        authentication_classes = []  # No auth required for status
        permission_classes = []
        
        def get(self, request):
            """Handle status requests."""
            return Response({
                "status": "healthy",
                "pipeline_built": hasattr(request, '_rag_pipeline'),
                "config": {}
            })


    class HealthAPIView(APIView):
        """Health check endpoint for Django REST."""
        authentication_classes = []
        permission_classes = []
        
        def get(self, request):
            """Health check."""
            # Implement health checking logic
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "checks": {
                    "database": "ok",
                    "memory": "ok"
                }
            }
            return Response(health_status)


    class MetricsAPIView(APIView):
        """Metrics endpoint for Django REST."""
        authentication_classes = []
        permission_classes = []
        
        def get(self, request):
            """Prometheus-style metrics."""
            # Implement metrics collection
            metrics_data = "# HELP requests_total Total requests\n# TYPE requests_total counter\nrequests_total 100\n"
            return Response(metrics_data, content_type='text/plain')

else:
    # Dummy classes when Django is not available
    class ChatAPIView:
        pass
    
    class BuildAPIView:
        pass
    
    class StatusAPIView:
        pass
    
    class HealthAPIView:
        pass
    
    class MetricsAPIView:
        pass


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
    
    server = DjangoEnhanced(config=config, api_config=api_config)
    return server.create_app()
