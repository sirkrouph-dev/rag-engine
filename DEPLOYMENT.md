# RAG Engine Production Deployment Guide

## Overview

The RAG Engine is now production-ready with support for multiple API frameworks (FastAPI, Flask, Django REST), web UIs (Streamlit, Gradio), and comprehensive deployment options.

## Key Features ✨

### 🏗️ **Multi-Framework Architecture**
- **FastAPI**: High-performance async API with automatic OpenAPI docs
- **Flask**: Lightweight and flexible web framework
- **Django REST**: Enterprise-grade framework with built-in admin
- **Seamless switching** between frameworks using `--framework` flag

### 🎨 **Web UI Options**
- **Streamlit**: Interactive data apps with built-in analytics
- **Gradio**: Simple chat interface for model interaction
- **Configurable** UI selection using `--ui` flag

### ⚙️ **Production Features**
- **Lazy pipeline initialization** for faster startup
- **Health checks** and status monitoring
- **CORS support** for cross-origin requests
- **Error handling** and graceful degradation
- **Configurable logging** and metrics
- **Docker containerization** ready
- **Load balancing** with Nginx

## Quick Start 🚀

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACE_API_TOKEN="your-hf-token"
```

### 3. Start API Server
```bash
# FastAPI (recommended for production)
python -m rag_engine serve --config config/production.json --framework fastapi --port 8000

# Flask (lightweight alternative)
python -m rag_engine serve --config config/production.json --framework flask --port 8001

# With Streamlit UI
python -m rag_engine serve --config config/production.json --framework fastapi --ui streamlit
```

### 4. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Build pipeline
curl -X POST http://localhost:8000/build

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?"}'
```

## Docker Deployment 🐳

### Single Container
```bash
# Build image
docker build -t rag-engine .

# Run FastAPI
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" rag-engine

# Run with custom config
docker run -p 8000:8000 -v ./config:/app/config rag-engine
```

### Multi-Service with Docker Compose
```bash
# Start all services
docker-compose up -d

# Scale specific services
docker-compose up --scale rag-engine-fastapi=3

# View logs
docker-compose logs -f rag-engine-fastapi
```

## API Documentation 📚

### Available Endpoints

#### **Health & Monitoring**
- `GET /` - Root endpoint with API info
- `GET /health` - Health check (always returns healthy)
- `GET /status` - System status and configuration

#### **Pipeline Management**
- `POST /build` - Build/rebuild the RAG pipeline
- `GET /pipeline/info` - Pipeline information and stats

#### **Chat & Query**
- `POST /chat` - Interactive chat with the RAG system
- `POST /query` - Direct query to the knowledge base

#### **Administrative**
- `GET /docs` - OpenAPI documentation (FastAPI only)
- `GET /redoc` - ReDoc documentation (FastAPI only)

### Request/Response Examples

#### Chat Request
```json
{
  "query": "What are the main features of this system?",
  "session_id": "user123",
  "top_k": 5
}
```

#### Chat Response
```json
{
  "query": "What are the main features of this system?",
  "response": "The system features multi-framework support...",
  "sources": [
    {
      "content": "Relevant document excerpt...",
      "score": 0.89,
      "metadata": {"source": "docs/features.md"}
    }
  ],
  "session_id": "user123",
  "status": "success"
}
```

## Configuration 🔧

### Production Configuration
The production config includes:
- **Security settings** (rate limiting, auth)
- **Performance optimizations** (caching, async processing)
- **Monitoring** (logging, metrics)
- **Scalability** (batch processing, connection pooling)

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_API_TOKEN=your-hf-token

# Optional
RAG_CONFIG_PATH=/path/to/config.json
RAG_LOG_LEVEL=INFO
RAG_CACHE_ENABLED=true
RAG_ASYNC_PROCESSING=true
```

## CLI Commands 🖥️

### Project Management
```bash
# Initialize new project
python -m rag_engine init my-rag-project --template advanced

# Build pipeline
python -m rag_engine build --config config.json

# Interactive chat
python -m rag_engine chat --config config.json
```

### Server Management
```bash
# Start with specific framework
python -m rag_engine serve --framework fastapi --port 8000

# Start with UI
python -m rag_engine serve --ui streamlit --ui-port 8501

# Development mode with auto-reload
python -m rag_engine serve --reload --framework fastapi
```

## Monitoring & Observability 📊

### Health Checks
All deployments include health check endpoints that monitor:
- API server responsiveness
- Pipeline build status
- Configuration validity
- Resource utilization

### Logging
Structured logging with configurable levels:
- **INFO**: General operational messages
- **WARNING**: Performance or configuration issues
- **ERROR**: System errors and failures
- **DEBUG**: Detailed troubleshooting information

### Metrics (Future Enhancement)
- Request/response times
- Pipeline build duration
- Cache hit rates
- Error rates by endpoint

## Scaling & Performance 🚀

### Horizontal Scaling
- **Load balancing** with Nginx
- **Container orchestration** with Docker Compose/Kubernetes
- **Database scaling** with vector store clustering

### Vertical Scaling
- **Async processing** for I/O operations
- **Batch embedding** for efficiency
- **Connection pooling** for database access
- **Memory optimization** with lazy loading

## Security Considerations 🔒

### API Security
- **Rate limiting** to prevent abuse
- **CORS configuration** for web integration
- **API key authentication** (configurable)
- **Input validation** and sanitization

### Deployment Security
- **Environment variable** management
- **Secret rotation** for API keys
- **Network isolation** with Docker networks
- **SSL/TLS termination** with Nginx

## Testing 🧪

### Automated Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=rag_engine tests/
```

### Manual Testing
```bash
# Test API endpoints
python test_api_endpoints.py

# Test pipeline functionality
python test_pipeline_simple.py

# Test server creation
python test_rag_api_creation.py
```

## Troubleshooting 🔍

### Common Issues

1. **Server won't start**
   - Check port availability: `netstat -ano | findstr :8000`
   - Verify config file syntax
   - Check environment variables

2. **Pipeline build fails**
   - Verify API keys are set
   - Check document paths exist
   - Validate configuration schema

3. **Chat queries fail**
   - Ensure pipeline is built: `POST /build`
   - Check LLM provider status
   - Verify embedding model availability

### Debug Mode
```bash
# Enable debug logging
export RAG_LOG_LEVEL=DEBUG

# Run with verbose output
python -m rag_engine serve --config config.json --log-level debug
```

## Future Enhancements 🔮

### Planned Features
- **Authentication & Authorization** system
- **Multi-tenant** support
- **Advanced caching** with Redis
- **Metrics & Analytics** dashboard
- **Model fine-tuning** integration
- **Kubernetes** deployment manifests

### Plugin System
- **Custom embedders** and LLM providers
- **Document processors** for new file types
- **Vector store** backends
- **UI themes** and customizations

---

## Support & Contributing 🤝

For issues, feature requests, or contributions, please visit our repository and follow the contribution guidelines.

**The RAG Engine is now production-ready with enterprise-grade features, multi-framework support, and comprehensive deployment options!** 🎉
