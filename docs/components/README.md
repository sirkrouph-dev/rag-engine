# RAG Engine Components

This directory contains documentation for each component type in the RAG Engine. Each component is designed to be modular, swappable, and configurable through the orchestration layer.

## Component Types

### Core Pipeline Components
- [**Loaders**](loaders.md) - Document loading and parsing
- [**Chunkers**](chunkers.md) - Text segmentation strategies
- [**Embedders**](embedders.md) - Text embedding models
- [**Vector Stores**](vectorstores.md) - Vector database backends
- [**Retrievers**](retrievers.md) - Document retrieval methods
- [**LLMs**](llms.md) - Language model providers
- [**Prompters**](prompters.md) - Prompt templates and processing

### Advanced Components
- [**Tools**](tools.md) - External tool integration
- [**Reasoning**](reasoning.md) - Multi-step reasoning engines

## Component Registry

All components are automatically registered through the `ComponentRegistry` in `rag_engine.core.component_registry`. This enables:

- **Dynamic component discovery**
- **Configuration-driven selection**
- **Easy swapping without code changes**
- **Plugin-based extensibility**

## Creating Custom Components

Each component type inherits from a base class and implements specific methods:

```python
from rag_engine.core.base import BaseRetriever

class MyCustomRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your retriever
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Implement retrieval logic
        pass
```

See individual component documentation for detailed implementation guides.

## Configuration

Components are configured through the main configuration file:

```json
{
  "retriever": {
    "type": "similarity",
    "config": {
      "top_k": 5,
      "score_threshold": 0.7
    }
  }
}
```

Each component's configuration options are documented in its respective page.
