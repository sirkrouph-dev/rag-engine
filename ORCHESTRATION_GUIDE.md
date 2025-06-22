# Modular Orchestration Layer

The RAG Engine now features a completely modular orchestration layer that decouples the API layer from specific component implementations. This allows you to easily swap different retrievers, LLMs, embedders, and other components without changing your API code.

## Overview

The modular orchestration layer consists of:

1. **Component Registry** - Manages different implementations of RAG components
2. **Orchestrator Interface** - Abstract interface for different RAG strategies  
3. **Factory Pattern** - Creates components based on configuration
4. **Plugin Architecture** - Allows extending with new backends

## Key Benefits

- **Loose Coupling**: API layer is independent of specific component implementations
- **Easy Swapping**: Change retrievers, LLMs, embedders without code changes
- **Extensibility**: Add new component types and orchestration strategies
- **Configuration-Driven**: Everything controlled through config files
- **Plugin Support**: Load custom components and orchestrators

## Architecture

### Component Registry

The `ComponentRegistry` manages all available component implementations:

```python
from rag_engine.core.orchestration import get_global_registry

registry = get_global_registry()

# List all available components
components = registry.list_components()
print(components)
# {
#   'loader': ['txt', 'pdf', 'docx', 'html'],
#   'chunker': ['fixed_size', 'sentence', 'token'],
#   'embedder': ['huggingface', 'openai', 'local'],
#   'vectorstore': ['chroma', 'faiss', 'pinecone'],
#   'retriever': ['similarity', 'bm25', 'hybrid', 'mmr'],
#   'llm': ['openai', 'anthropic', 'local', 'ollama'],
#   'prompter': ['default', 'conversational', 'qa', 'summarization']
# }

# Create a component instance
embedder = registry.create_component('embedder', 'huggingface', {'model': 'all-MiniLM-L6-v2'})
```

### Orchestrators

Orchestrators define different RAG strategies and how components work together:

#### Default Orchestrator
The standard RAG pipeline: load → chunk → embed → store → retrieve → prompt → generate

#### Hybrid Retrieval Orchestrator  
Combines multiple retrieval methods (semantic + BM25) for better results

#### Multi-Modal Orchestrator
Handles multiple content types (text, images, etc.)

### Using Different Orchestrators

#### Via CLI
```bash
# Use default orchestrator
python -m rag_engine serve --config config.json --orchestrator default

# Use hybrid retrieval
python -m rag_engine serve --config config.json --orchestrator hybrid

# Use multi-modal
python -m rag_engine serve --config config.json --orchestrator multimodal
```

#### Via API
```python
from rag_engine.interfaces.base_api import BaseAPIServer
from rag_engine.config.loader import load_config

config = load_config('config.json')

# Create API server with hybrid orchestrator
server = FastAPIServer(config=config, orchestrator_type="hybrid")
app = server.create_app()
```

#### Programmatically
```python
from rag_engine.core.orchestration import create_orchestrator
from rag_engine.config.loader import load_config

config = load_config('config.json')

# Create different orchestrators
default_orch = create_orchestrator("default", config)
hybrid_orch = create_orchestrator("hybrid", config)
multimodal_orch = create_orchestrator("multimodal", config)

# Build and use
hybrid_orch.build()
result = hybrid_orch.query("What is the main topic?")
```

## Component Swapping Examples

### Different Retrievers

You can easily switch between different retrieval methods by changing the config:

```json
{
  "retrieval": {
    "method": "similarity",  // or "bm25", "hybrid", "mmr"
    "top_k": 5
  }
}
```

### Different LLMs

Switch between LLM providers without code changes:

```json
{
  "llm": {
    "provider": "openai",     // or "anthropic", "local", "ollama"
    "model": "gpt-4",
    "temperature": 0.7
  }
}
```

### Different Vector Stores

Use different vector databases:

```json
{
  "vectorstore": {
    "provider": "chroma",     // or "faiss", "pinecone"
    "persist_directory": "./vectors"
  }
}
```

## Creating Custom Orchestrators

You can create custom orchestrators for specialized RAG strategies:

```python
from rag_engine.core.orchestration import BaseOrchestrator, OrchestratorFactory

class CustomOrchestrator(BaseOrchestrator):
    \"\"\"Custom orchestrator with specialized logic.\"\"\"
    
    def build(self) -> None:
        \"\"\"Build custom pipeline.\"\"\"
        # Custom component initialization
        self._create_specialized_components()
        self._is_built = True
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        \"\"\"Custom query processing.\"\"\"
        # Implement custom query logic
        return {"answer": "Custom response", "status": "success"}
    
    def get_status(self) -> Dict[str, Any]:
        \"\"\"Return orchestrator status.\"\"\"
        return {
            "orchestrator": "CustomOrchestrator",
            "is_built": self._is_built,
            "components": list(self.components.keys())
        }

# Register the custom orchestrator
OrchestratorFactory.register_orchestrator("custom", CustomOrchestrator)
```

## Adding Custom Components

Register new component implementations:

```python
from rag_engine.core.orchestration import get_global_registry

class CustomRetriever:
    def retrieve(self, query, **kwargs):
        # Custom retrieval logic
        return results

# Register the custom component
registry = get_global_registry()
registry.register_component(
    'retriever',
    'custom_retriever',
    CustomRetriever,
    "Custom retrieval implementation"
)
```

## API Endpoints for Orchestrator Management

The enhanced API includes endpoints for managing orchestrators:

### Get Orchestrator Status
```bash
GET /orchestrator/status
```

Returns current orchestrator information and component status.

### List Available Components  
```bash
GET /orchestrator/components
```

Returns all registered component types and implementations.

### Rebuild Orchestrator
```bash
POST /orchestrator/rebuild  
```

Rebuilds the orchestrator with the current configuration.

### Get Component Status
```bash
GET /orchestrator/components/{component_type}
```

Returns status of a specific component type.

## Configuration Examples

### Basic Configuration
```json
{
  "documents": [{"path": "docs/", "type": "directory"}],
  "chunking": {
    "method": "fixed_size",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_tokens": 100,
    "overlap": 50
  },
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "./vector_db"
  },
  "retrieval": {
    "method": "similarity",
    "top_k": 5
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7
  },
  "prompting": {
    "template": "default",
    "system_prompt": "You are a helpful assistant."
  },
  "output": {
    "method": "direct"
  }
}
```

### Hybrid Retrieval Configuration
```json
{
  "retrieval": {
    "method": "hybrid",
    "semantic_weight": 0.7,
    "bm25_weight": 0.3,
    "top_k": 10,
    "rerank": true
  }
}
```

## Benefits for Different Use Cases

### Research Applications
- Use hybrid retrieval for comprehensive document discovery
- Swap between different embedding models for domain-specific tasks
- Use multi-modal orchestrator for documents with images

### Production Systems  
- Easy A/B testing of different retrieval methods
- Hot-swap LLM providers for cost optimization
- Scale different components independently

### Development
- Test different configurations without code changes
- Rapid prototyping of new RAG strategies
- Easy debugging of individual components

## Migration from Hardcoded Pipeline

If you're upgrading from the previous hardcoded pipeline:

1. **No breaking changes** - existing code continues to work
2. **Gradual migration** - can adopt orchestrators incrementally  
3. **Configuration compatibility** - existing configs work with orchestrators
4. **API compatibility** - all existing endpoints remain functional

## Best Practices

1. **Start with Default** - Use the default orchestrator for most use cases
2. **Test Orchestrators** - Try different orchestrators with your data
3. **Monitor Performance** - Use `/orchestrator/status` to monitor components
4. **Configuration Management** - Use environment variables for sensitive config
5. **Custom Components** - Create custom components for specialized needs

## Troubleshooting

### Component Not Found
If you get "Component not found" errors:

```python
# Check available components
from rag_engine.core.orchestration import get_global_registry
registry = get_global_registry()
print(registry.list_components())
```

### Orchestrator Build Failures
Check orchestrator status:

```bash
GET /orchestrator/status
```

### Configuration Errors
Validate your configuration against the schema:

```python
from rag_engine.config.loader import load_config
config = load_config('config.json')  # Will validate and show errors
```

The modular orchestration layer makes the RAG Engine highly flexible and extensible while maintaining simplicity for common use cases.
