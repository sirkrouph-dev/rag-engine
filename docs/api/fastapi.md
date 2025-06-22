# FastAPI Implementation

The FastAPI implementation provides a high-performance, production-ready API with automatic documentation, type validation, and async support.

## Features

- **Automatic Documentation**: Swagger UI and ReDoc
- **Type Safety**: Pydantic models for request/response validation
- **Async Support**: High-concurrency request handling
- **Multi-worker Scaling**: Production deployment support
- **CORS Middleware**: Cross-origin request support
- **Background Tasks**: Non-blocking operations

## Configuration

### Basic Setup
```json
{
  "api": {
    "framework": "fastapi",
    "config": {
      "host": "0.0.0.0",
      "port": 8000,
      "workers": 1,
      "reload": true,
      "log_level": "info"
    }
  }
}
```

### Production Configuration
```json
{
  "api": {
    "framework": "fastapi",
    "config": {
      "host": "0.0.0.0",
      "port": 8000,
      "workers": 4,
      "reload": false,
      "log_level": "warning",
      "access_log": true,
      "cors_origins": ["https://myapp.com"],
      "docs_url": "/docs",
      "redoc_url": "/redoc"
    }
  }
}
```

## Usage Examples

### Starting the Server
```bash
# Development mode
python -m rag_engine serve --api fastapi --host 0.0.0.0 --port 8000

# Production mode with multiple workers
python -m rag_engine serve --api fastapi --workers 4 --no-reload

# With custom config
python -m rag_engine serve --config config.json --api fastapi
```

### API Endpoints

The FastAPI server provides several endpoint categories:

#### Health and Status
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /status` - System status and configuration

#### Pipeline Management
- `POST /build` - Build the RAG pipeline
- `GET /config` - Get current configuration (sanitized)

#### Chat and Query
- `POST /chat` - Chat with the RAG system
- `POST /query` - Direct query endpoint (alias for chat)

#### Document Management
- `GET /documents` - List configured documents
- `GET /chunks` - List document chunks

#### Orchestrator Management
- `GET /orchestrator/status` - Get orchestrator status
- `GET /orchestrator/components` - List available components
- `POST /orchestrator/rebuild` - Rebuild orchestrator
- `GET /orchestrator/components/{component_type}` - Get component status

### Request/Response Models

#### Chat Request
```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
```

#### Chat Response
```python
class ChatResponse(BaseModel):
    query: str
    response: str
    session_id: Optional[str] = None
    status: str
    metadata: Optional[Dict[str, Any]] = None
```

## Advanced Features

### Middleware Configuration
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Authentication
```json
{
  "api": {
    "config": {
      "authentication": {
        "enabled": true,
        "type": "bearer_token",
        "secret_key": "${API_SECRET_KEY}",
        "token_expire_hours": 24
      }
    }
  }
}
```

### Rate Limiting
```json
{
  "api": {
    "config": {
      "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 60,
        "burst_size": 10
      }
    }
  }
}
```

### Background Tasks
```python
from fastapi import BackgroundTasks

@app.post("/build")
async def build_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(rebuild_vector_index)
    return {"message": "Build started", "status": "processing"}
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_engine.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CONFIG_PATH=/app/config/production.json
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    command: ["uvicorn", "rag_engine.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: rag-engine:latest
        ports:
        - containerPort: 8000
        env:
        - name: RAG_CONFIG_PATH
          value: "/app/config/production.json"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring and Observability

### Metrics Integration
```json
{
  "api": {
    "config": {
      "metrics": {
        "enabled": true,
        "endpoint": "/metrics",
        "include_request_metrics": true,
        "include_custom_metrics": true
      }
    }
  }
}
```

### Logging Configuration
```json
{
  "api": {
    "config": {
      "logging": {
        "level": "INFO",
        "format": "json",
        "include_request_id": true,
        "log_requests": true,
        "log_responses": false
      }
    }
  }
}
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "database": "healthy",
            "llm_service": "healthy",
            "vector_store": "healthy"
        }
    }
```

## Custom Extensions

### Adding Custom Endpoints
```python
from rag_engine.interfaces.api import FastAPIServer

class CustomFastAPIServer(FastAPIServer):
    def add_routes(self):
        super().add_routes()
        
        @self.app.get("/custom/search")
        async def custom_search(query: str, filters: Optional[str] = None):
            # Custom search logic
            return {"results": []}
        
        @self.app.post("/custom/upload")
        async def upload_document(file: UploadFile):
            # Document upload logic
            return {"message": "Document uploaded"}
```

### WebSocket Support
```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process chat message
            response = await process_chat(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("Client disconnected")
```

## Performance Optimization

### Connection Pooling
```json
{
  "api": {
    "config": {
      "connection_pool": {
        "max_connections": 100,
        "max_keepalive_connections": 20,
        "keepalive_expiry": 30
      }
    }
  }
}
```

### Caching
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Initialize caching
FastAPICache.init(RedisBackend(), prefix="rag-cache")

@app.get("/cached-endpoint")
@cache(expire=3600)  # Cache for 1 hour
async def cached_endpoint():
    return {"data": "expensive_computation"}
```

### Response Compression
```json
{
  "api": {
    "config": {
      "compression": {
        "enabled": true,
        "minimum_size": 1000,
        "compression_level": 6
      }
    }
  }
}
```

## Testing

### Unit Tests
```python
from fastapi.testclient import TestClient
from rag_engine.interfaces.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    response = client.post(
        "/chat",
        json={"query": "What is AI?", "session_id": "test"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

### Integration Tests
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_full_pipeline():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Build pipeline
        build_response = await client.post("/build")
        assert build_response.status_code == 200
        
        # Test chat
        chat_response = await client.post(
            "/chat",
            json={"query": "Test question"}
        )
        assert chat_response.status_code == 200
```

## Security

### HTTPS Configuration
```json
{
  "api": {
    "config": {
      "ssl": {
        "enabled": true,
        "cert_file": "/path/to/cert.pem",
        "key_file": "/path/to/key.pem"
      }
    }
  }
}
```

### Input Validation
```python
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v
    
    @validator('query')
    def query_length_limit(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        return v
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Use different port
python -m rag_engine serve --api fastapi --port 8001
```

**2. CORS Issues**
```json
{
  "api": {
    "config": {
      "cors": {
        "allow_origins": ["http://localhost:3000", "https://myapp.com"],
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["*"],
        "allow_credentials": true
      }
    }
  }
}
```

**3. Memory Issues with Multiple Workers**
```json
{
  "api": {
    "config": {
      "workers": 2,
      "max_requests": 1000,
      "max_requests_jitter": 100,
      "preload_app": true
    }
  }
}
```

**4. Slow Response Times**
```json
{
  "api": {
    "config": {
      "performance": {
        "timeout": 30,
        "keep_alive": 2,
        "max_concurrent_requests": 100
      }
    }
  }
}
```

## Dependencies

```bash
# Core FastAPI
pip install fastapi uvicorn[standard]

# Optional enhancements
pip install python-multipart  # File uploads
pip install python-jose[cryptography]  # JWT tokens
pip install passlib[bcrypt]  # Password hashing
pip install aiofiles  # Async file operations
pip install fastapi-cache[redis]  # Caching
pip install prometheus-fastapi-instrumentator  # Metrics
```
