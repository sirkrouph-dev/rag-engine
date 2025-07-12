# RAG Engine - AI Coding Agent Instructions

## Project Overview
This is a **production-ready, modular RAG (Retrieval-Augmented Generation) engine** with enterprise-grade security, monitoring, and deployment capabilities. The architecture emphasizes configurability, component swapping, and orchestration patterns.

## Core Architecture Patterns

### 1. **Component Registry & Orchestration System**
- **Central Pattern**: All RAG components (embedders, retrievers, LLMs, etc.) are registered via `ComponentRegistry` in `rag_engine/core/orchestration.py`
- **Key Insight**: Never hardcode component implementations - always use the registry for dynamic component creation
- **Usage**: `registry.create_component('embedder', 'huggingface', config)` instead of direct class instantiation
- **Orchestrators**: Use `DefaultOrchestrator`, `HybridOrchestrator`, or create custom orchestration strategies

### 2. **Configuration-Driven Development**
- **Schema Location**: `rag_engine/config/schema.py` - Pydantic models define all configuration contracts
- **Config Loading**: Always use `load_config()` from `rag_engine/config/loader.py` - handles validation and environment variable substitution
- **Environment Variables**: Use `${VAR_NAME:-default}` pattern in JSON configs for environment-aware settings
- **Hierarchical Config**: `Global → API → Orchestrator → Components` - understand this hierarchy when debugging

### 3. **Production Infrastructure**
- **Production API**: Use `rag_engine/interfaces/production_api.py` - includes security, monitoring, circuit breakers
- **Security Layer**: JWT auth, rate limiting, input validation in `rag_engine/core/security.py`
- **Monitoring**: Prometheus metrics, health checks in `rag_engine/core/monitoring.py`
- **Reliability**: Circuit breakers, retry logic in `rag_engine/core/reliability.py`

## Essential Development Workflows

### CLI Commands (Primary Interface)
```bash
# Build pipeline from config
python -m rag_engine build --config configs/production.json

# Start production server with orchestrator
python -m rag_engine serve --framework fastapi --orchestrator hybrid --workers 4

# Interactive chat with specific config
python -m rag_engine chat --config examples/configs/demo_local_config.json

# Initialize new project
python -m rag_engine init my-project --template advanced
```

### Component Development Pattern
```python
# Always register new components with the global registry
from rag_engine.core.orchestration import get_global_registry

@dataclass
class CustomRetriever:
    def retrieve(self, query: str, **kwargs) -> List[Document]:
        # Implementation
        pass

# Register with type, name, class, description
registry = get_global_registry()
registry.register_component('retriever', 'custom', CustomRetriever, "Custom retrieval logic")
```

### Testing Strategy
- **Unit Tests**: `tests/unit/` - component-level testing
- **Integration Tests**: `tests/integration/` - full pipeline testing with mocked external services
- **CLI Testing**: `tests/integration/test_cli.py` - comprehensive CLI command testing
- **Production Tests**: `test_production_readiness.py` - validates production infrastructure

## Docker & Deployment Patterns

### Development vs Production
- **Development**: `docker-compose.yml` - single services, local LLMs
- **Demo**: `docker-compose.demo.yml` - includes frontend, optimized for demos
- **Production**: `docker-compose.production.yml` - full stack with monitoring, security, scaling

### Critical Docker Commands
```bash
# Production deployment
docker-compose -f docker/docker-compose.production.yml up -d

# Development with auto-reload
docker-compose up -d && docker-compose logs -f rag-engine

# Scale specific services
docker-compose up -d --scale rag-engine=3

# Health check debugging
docker-compose exec rag-engine curl http://localhost:8000/health/live
```

## Configuration Conventions

### Stack Management
- **Preset Stacks**: `DEMO`, `LOCAL`, `CLOUD`, `MINI`, `FULL` - defined in requirements/stacks/
- **Stack Selection**: Use `scripts/ai_setup.py` for intelligent stack configuration
- **Bloat Management**: Install only required dependencies based on chosen components

### Environment Configuration
- **Development**: Use local configs in `examples/configs/`
- **Production**: Use `configs/production.json` with environment variable substitution
- **Security**: All secrets via environment variables, never hardcoded

## Frontend Integration

### Vue.js Architecture
- **API Service**: `frontend/src/services/api.js` - centralized backend communication
- **State Management**: Pinia stores in `frontend/src/stores/` - system and chat state
- **Component Pattern**: Reusable components in `frontend/src/components/` with Tailwind CSS
- **Build Process**: `npm run build` produces `dist/` for production deployment

### Key API Endpoints
```javascript
// Essential endpoints for frontend integration
await api.getHealth()              // System health
await api.buildPipeline()          // Build RAG pipeline  
await api.sendMessage(query)       // Chat functionality
await api.getOrchestratorStatus()  // Component status
await api.configureStack(type)     // Stack management
```

## Debugging & Monitoring

### Health Check Hierarchy
1. `/health/live` - Container liveness
2. `/health/ready` - Service readiness  
3. `/health/db` - Database connectivity
4. `/orchestrator/status` - Component health
5. `/metrics` - Prometheus metrics

### Log Analysis
- **Application Logs**: `logs/` directory or Docker logs
- **Component Errors**: Check orchestrator status first
- **Configuration Issues**: Validate with Pydantic schema
- **API Errors**: Check security middleware and rate limits

## Security Considerations

### Authentication Flow
- JWT tokens via `rag_engine/core/security.py`
- Rate limiting per endpoint (default: 100 req/min)
- Input validation prevents SQL injection and XSS
- Audit logging for all operations

### Production Security
- Use production configs with proper secrets management
- Enable HTTPS via nginx configuration
- Configure CORS for frontend integration
- Regular security updates for dependencies

## Integration Points

### External Services
- **LLM Providers**: OpenAI, Anthropic, local Ollama - configured via orchestrator
- **Vector Databases**: ChromaDB, Pinecone, FAISS - swappable via config
- **Monitoring**: Prometheus + Grafana stack in production
- **Caching**: Redis for response and computation caching

### Plugin Architecture
- Components auto-register on import
- Custom orchestrators extend `BaseOrchestrator`
- Plugin discovery via Python packaging and entry points
- Configuration-driven component selection

This is a **mature, production-ready framework** - prioritize understanding the orchestration patterns and configuration system for maximum productivity.
