# Enhanced API Framework Customization Guide

The RAG Engine now includes a comprehensive customizable API framework system that supports FastAPI, Flask, and Django REST with advanced features for production deployment.

## Features Overview

### 🚀 Multi-Framework Support
- **FastAPI Enhanced**: Production-ready with async support, automatic documentation, and high performance
- **Flask Enhanced**: Lightweight and flexible with comprehensive middleware support
- **Django REST Enhanced**: Enterprise-grade with full ORM integration and admin interface

### 🔒 Security & Authentication
- **Multiple Auth Methods**: API Key, JWT, OAuth2, Basic Auth
- **Rate Limiting**: Per-IP, per-user, per-endpoint, or global limits
- **CORS Configuration**: Customizable cross-origin resource sharing
- **Security Headers**: Automatic security header injection
- **IP Filtering**: Allow/block specific IP addresses

### 📊 Monitoring & Observability
- **Metrics Collection**: Prometheus-compatible metrics endpoint
- **Health Checks**: Comprehensive health and readiness probes
- **Request Logging**: Detailed request/response logging
- **Performance Tracking**: Response time and throughput monitoring

### ⚡ Performance & Scaling
- **Multi-Worker Support**: Production scaling with multiple workers
- **Response Caching**: Intelligent caching with TTL configuration
- **Compression**: Automatic gzip compression for large responses
- **Connection Pooling**: Optimized database and API connections

### 🔧 Customization Options
- **Custom Middleware**: Plugin-based middleware system
- **Custom Routes**: Dynamic route registration
- **Error Handling**: Configurable error responses
- **Request/Response Transformers**: Custom data processing

## Quick Start

### 1. Basic Configuration

Create a configuration file `config/enhanced.json`:

```json
{
  "api": {
    "framework": "fastapi",
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "debug": false,
    "enable_docs": true,
    "enable_metrics": true,
    "enable_health_checks": true
  },
  "security": {
    "auth_method": "api_key",
    "api_keys": ["your-api-key-here"],
    "enable_rate_limiting": true,
    "requests_per_minute": 100,
    "cors_origins": ["https://yourdomain.com"]
  },
  "rag": {
    "documents": [{"type": "pdf", "path": "./documents"}],
    "chunking": {"method": "recursive", "max_tokens": 512},
    "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    "vectorstore": {"provider": "chroma", "persist_directory": "./vector_store"},
    "llm": {"provider": "openai", "model": "gpt-4"}
  }
}
```

### 2. Start the Enhanced Server

```bash
# Using CLI with enhanced framework
rag-engine serve --config config/enhanced.json --framework fastapi --workers 4

# Or with specific customization
rag-engine serve --config config/enhanced.json --framework flask --port 5000 --workers 2
```

### 3. Programmatic Usage

```python
from rag_engine.interfaces.enhanced_base_api import enhanced_factory, APICustomization, AuthMethod
from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced

# Create custom API configuration
api_config = APICustomization(
    host="0.0.0.0",
    port=8000,
    workers=4,
    debug=False,
    enable_docs=True,
    enable_metrics=True,
    enable_health_checks=True,
    enable_rate_limiting=True,
    auth_method=AuthMethod.API_KEY,
    api_keys=["your-secure-api-key"],
    custom_headers={"X-Powered-By": "RAG-Engine-Enhanced"}
)

# Register and create server
enhanced_factory.register_framework("fastapi", FastAPIEnhanced)
server = enhanced_factory.create_server("fastapi", api_config=api_config)

# Start the server
server.start_server()
```

## Configuration Reference

### API Configuration (`APICustomization`)

```python
api_config = APICustomization(
    # Server Settings
    host="0.0.0.0",                    # Server host
    port=8000,                         # Server port
    workers=1,                         # Number of worker processes
    debug=False,                       # Debug mode
    reload=False,                      # Auto-reload on code changes
    
    # Documentation
    enable_docs=True,                  # Enable API documentation
    docs_url="/docs",                  # Documentation URL
    openapi_url="/openapi.json",       # OpenAPI schema URL
    
    # Security
    auth_method=AuthMethod.NONE,       # Authentication method
    api_keys=[],                       # Valid API keys
    jwt_secret=None,                   # JWT signing secret
    cors_origins=["*"],                # CORS allowed origins
    
    # Rate Limiting
    enable_rate_limiting=False,        # Enable rate limiting
    rate_limit_type=RateLimitType.PER_IP,  # Rate limit strategy
    requests_per_minute=60,            # Requests per minute limit
    burst_size=10,                     # Burst request limit
    
    # Middleware
    enable_compression=True,           # Enable gzip compression
    enable_request_logging=True,       # Log all requests
    enable_response_caching=False,     # Cache responses
    cache_ttl=300,                     # Cache time-to-live (seconds)
    
    # Error Handling
    include_error_details=False,       # Include error details in responses
    custom_error_handlers={},          # Custom error handlers
    
    # Monitoring
    enable_metrics=False,              # Enable metrics collection
    metrics_endpoint="/metrics",       # Metrics endpoint URL
    enable_health_checks=True,         # Enable health checks
    
    # Customization
    custom_headers={},                 # Custom response headers
    custom_routes=[],                  # Additional routes
    middleware_stack=[]                # Custom middleware stack
)
```

### Security Configuration

```python
from rag_engine.interfaces.security import SecurityConfig

security = SecurityConfig(
    # Authentication
    enable_auth=True,
    auth_method="jwt",                 # api_key, jwt, oauth2, basic
    api_keys=["key1", "key2"],
    jwt_secret="your-secret-key",
    jwt_algorithm="HS256",
    jwt_expiry_seconds=3600,
    
    # Rate Limiting
    enable_rate_limiting=True,
    rate_limit_per_minute=100,
    rate_limit_burst=20,
    rate_limit_storage="memory",       # memory, redis
    
    # CORS
    cors_origins=["https://yourdomain.com"],
    cors_methods=["GET", "POST", "PUT", "DELETE"],
    cors_headers=["*"],
    
    # Security Headers
    enable_security_headers=True,
    security_headers={
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block"
    },
    
    # IP Filtering
    allowed_ips=["192.168.1.0/24"],
    blocked_ips=["10.0.0.0/8"]
)
```

## Advanced Features

### Custom Middleware

```python
def custom_logging_middleware(request, call_next):
    """Custom request logging middleware."""
    start_time = time.time()
    response = call_next(request)
    duration = time.time() - start_time
    
    logger.info(f"Request processed in {duration:.4f}s")
    return response

# Register middleware
server.register_middleware("logging", custom_logging_middleware)
```

### Custom Routes

```python
def custom_analytics_endpoint():
    """Custom analytics endpoint."""
    return {
        "total_requests": metrics.total_requests,
        "avg_response_time": metrics.avg_response_time,
        "active_users": metrics.active_users
    }

# Add custom route
server.add_route("/analytics", custom_analytics_endpoint, ["GET"])
```

### Plugin System

```python
from rag_engine.interfaces.plugins import CachePlugin, LoggingPlugin

# Create server with plugins
server = enhanced_factory.create_server(
    "fastapi", 
    api_config=api_config,
    plugins=["cache", "logging", "custom_analytics"]
)
```

## Production Deployment

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use enhanced production configuration
ENV RAG_CONFIG_PATH=config/enhanced_production.json

# Start with multiple workers
CMD ["rag-engine", "serve", "--config", "$RAG_CONFIG_PATH", "--framework", "fastapi", "--workers", "4"]
```

### Docker Compose with Monitoring

```yaml
version: '3.8'
services:
  rag-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CONFIG_PATH=config/enhanced_production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./documents:/app/documents
      - ./vector_store:/app/vector_store
    command: ["rag-engine", "serve", "--config", "config/enhanced_production.json", "--framework", "fastapi", "--workers", "4"]
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine-enhanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-engine
  template:
    metadata:
      labels:
        app: rag-engine
    spec:
      containers:
      - name: rag-engine
        image: rag-engine:enhanced
        ports:
        - containerPort: 8000
        env:
        - name: RAG_CONFIG_PATH
          value: "config/enhanced_production.json"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-engine-service
spec:
  selector:
    app: rag-engine
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring & Observability

### Available Endpoints

- `GET /health` - Health check
- `GET /health/ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /status` - System status
- `GET /docs` - API documentation (if enabled)

### Metrics Collection

The enhanced frameworks automatically collect:

- Request count by endpoint and method
- Response time distribution
- Error rate by status code
- Active connections
- Memory and CPU usage
- Cache hit/miss ratios

### Grafana Dashboard

Import the provided Grafana dashboard configuration to visualize:

- Request throughput
- Response time percentiles
- Error rate trends
- Resource utilization
- Cache performance

## Security Best Practices

1. **Always use HTTPS in production**
2. **Rotate API keys regularly**
3. **Implement proper CORS policies**
4. **Enable rate limiting**
5. **Monitor for suspicious activity**
6. **Keep dependencies updated**
7. **Use strong JWT secrets**
8. **Implement proper logging**

## Troubleshooting

### Common Issues

1. **High memory usage**: Adjust cache settings and worker count
2. **Slow responses**: Enable compression and response caching
3. **Authentication failures**: Check API key configuration
4. **Rate limit errors**: Adjust rate limiting settings
5. **Framework not loading**: Verify framework dependencies are installed

### Debug Mode

Enable debug mode for detailed error information:

```python
api_config = APICustomization(
    debug=True,
    include_error_details=True,
    enable_request_logging=True
)
```

## Framework-Specific Notes

### FastAPI Enhanced
- Supports async operations
- Automatic API documentation
- Best performance for high-throughput scenarios
- Built-in validation with Pydantic

### Flask Enhanced
- Lightweight and flexible
- Easy to extend with custom middleware
- Good for microservices
- Extensive ecosystem of extensions

### Django REST Enhanced
- Full ORM integration
- Admin interface included
- Enterprise-grade features
- Built-in user management

Choose the framework that best fits your project requirements and team expertise.

## Custom Server Support

The RAG Engine now supports **custom user-defined servers** that aren't officially supported but can integrate with the RAG framework. This allows you to use any web framework while still getting RAG functionality.

### 🔧 Creating Custom Servers

#### 1. Generate a Template

```bash
# Create a custom server template
rag-engine custom-server create --name MyAPI --framework tornado

# This creates myapi_server.py with a template
```

#### 2. Implement Your Server

```python
"""
Custom RAG Engine server: MyAPI
"""
from typing import Dict, Any, Optional
from rag_engine.interfaces.custom_servers import CustomServerBase
import tornado.web
import tornado.ioloop
import json


class MyapiServer(CustomServerBase):
    """Custom RAG Engine server using tornado."""
    
    def __init__(self, config: Optional[Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.app = None
    
    def create_app(self):
        """Create Tornado application with RAG endpoints."""
        
        class BaseHandler(tornado.web.RequestHandler):
            def initialize(self, server):
                self.server = server
        
        class ChatHandler(BaseHandler):
            def post(self):
                data = json.loads(self.request.body)
                result = self.server.rag_endpoints['chat'](
                    data.get('query'), 
                    data.get('session_id')
                )
                self.write(result)
        
        class BuildHandler(BaseHandler):
            def post(self):
                result = self.server.rag_endpoints['build']()
                self.write(result)
        
        self.app = tornado.web.Application([
            (r"/chat", ChatHandler, dict(server=self)),
            (r"/build", BuildHandler, dict(server=self)),
            (r"/status", lambda: self.rag_endpoints['status']()),
            (r"/health", lambda: self.rag_endpoints['health']()),
        ])
        
        return self.app
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start Tornado server."""
        if not self.app:
            self.create_app()
        
        print(f"🌪️  Starting Tornado custom server on http://{host}:{port}")
        self.app.listen(port, address=host)
        tornado.ioloop.IOLoop.current().start()


# Register your custom server
from rag_engine.interfaces.enhanced_base_api import enhanced_factory

enhanced_factory.register_custom_server(
    "myapi",
    MyapiServer,
    "Custom server using tornado"
)
```

#### 3. Use Your Custom Server

```bash
# Validate your implementation
rag-engine custom-server validate --file myapi_server.py

# Start your custom server
rag-engine serve --config config.json --framework myapi --port 8000
```

### 📋 Available Custom Server Commands

```bash
# List all available servers (built-in and custom)
rag-engine custom-server list

# Create a new custom server template
rag-engine custom-server create --name MyServer --framework flask

# Validate a custom server implementation
rag-engine custom-server validate --file my_server.py

# Register a custom server module
rag-engine custom-server register --name myserver --file my_server.py
```

### 🏗️ Custom Server Interface

All custom servers must implement:

```python
class MyCustomServer(CustomServerBase):
    def create_app(self):
        """Create and return your application instance."""
        pass
    
    def start_server(self, host="0.0.0.0", port=8000, **kwargs):
        """Start your server."""
        pass
```

### 🔌 Built-in RAG Integration

Custom servers automatically get access to RAG functionality:

```python
# Available RAG endpoints in self.rag_endpoints:
self.rag_endpoints['chat'](query, session_id)      # Chat with RAG
self.rag_endpoints['build']()                       # Build pipeline
self.rag_endpoints['status']()                      # Get status
self.rag_endpoints['health']()                      # Health check
```

### 📝 Example Implementations

The RAG Engine includes example custom servers for:

- **Tornado**: High-performance async server
- **Bottle**: Lightweight WSGI framework  
- **CherryPy**: Object-oriented web framework

```bash
# Install dependencies and use example servers
pip install tornado
rag-engine serve --framework tornado --config config.json

pip install bottle
rag-engine serve --framework bottle --config config.json
```

### ⚠️ Custom Server Limitations

- **Not Officially Supported**: Custom servers are user-maintained
- **Limited Features**: May not support all enhanced framework features
- **No Guaranteed Compatibility**: Updates may break custom implementations
- **Basic Integration**: Only core RAG endpoints are provided

### 🛠️ Advanced Custom Server Features

#### Optional Methods You Can Implement

```python
class AdvancedCustomServer(CustomServerBase):
    def add_rag_integration(self, rag_config):
        """Called when RAG config is available."""
        super().add_rag_integration(rag_config)
        # Custom integration logic
    
    def add_middleware(self, middleware_type, handler):
        """Add middleware if your framework supports it."""
        pass
    
    def add_route(self, path, handler, methods):
        """Add dynamic routes if supported."""
        pass
    
    def set_error_handler(self, status_code, handler):
        """Set custom error handlers if supported."""
        pass
```

#### Custom Error Handling

```python
def default_chat_handler(self, query: str, session_id: str = None):
    """Override default chat handler with custom logic."""
    try:
        # Your custom pre-processing
        query = self.preprocess_query(query)
        
        # Use the pipeline
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            return {"error": "Pipeline not built", "status": "error"}
        
        response = self.pipeline.query(query)
        
        # Your custom post-processing
        response = self.postprocess_response(response)
        
        return {
            "query": query,
            "response": response,
            "session_id": session_id,
            "status": "success",
            "server": "custom"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}
```
