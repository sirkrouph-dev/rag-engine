# Development Guide

This guide covers setting up a development environment, contribution guidelines, and development best practices for the RAG Engine.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, or poetry)

### Environment Setup

#### Clone Repository
```bash
git clone https://github.com/your-org/rag-engine.git
cd rag-engine
```

#### Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n rag-engine python=3.10
conda activate rag-engine

# Using poetry
poetry install
poetry shell
```

#### Install Dependencies
```bash
# Development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

### Development Dependencies

Create `requirements-dev.txt`:
```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code Quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=1.0.0

# Development Tools
ipython>=8.0.0
jupyter>=1.0.0
python-dotenv>=1.0.0
```

## Code Structure

### Project Layout
```
rag_engine/
├── __init__.py
├── __main__.py              # CLI entry point
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── loader.py           # Config loading logic
│   └── schema.py           # Pydantic schemas
├── core/                   # Core components
│   ├── __init__.py
│   ├── base.py             # Base classes
│   ├── orchestration.py   # Orchestration layer
│   ├── component_registry.py
│   ├── loader.py           # Document loaders
│   ├── chunker.py          # Text chunkers
│   ├── embedder.py         # Embedding models
│   ├── vectorstore.py      # Vector databases
│   ├── retriever.py        # Retrieval methods
│   ├── llm.py              # Language models
│   └── prompting.py        # Prompt management
├── interfaces/             # API interfaces
│   ├── __init__.py
│   ├── base_api.py         # Base API server
│   ├── api.py              # FastAPI implementation
│   └── cli.py              # CLI interface
└── plugins/                # Plugin system
    └── __init__.py
```

### Coding Standards

#### Python Style Guide
Follow PEP 8 with these specific guidelines:

```python
# Imports
import os
import sys
from typing import Dict, List, Optional, Union

from third_party_package import SomeClass

from rag_engine.core.base import BaseComponent
from rag_engine.config.schema import Config

# Type hints are required
def process_documents(
    documents: List[Dict[str, str]], 
    config: Config
) -> List[Dict[str, str]]:
    """Process documents with given configuration.
    
    Args:
        documents: List of document dictionaries
        config: Configuration object
        
    Returns:
        Processed documents
        
    Raises:
        ValueError: If documents are invalid
    """
    pass

# Class definitions
class CustomRetriever(BaseRetriever):
    """Custom retriever implementation.
    
    This retriever implements a specialized search algorithm
    for domain-specific document retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.algorithm = config.get("algorithm", "default")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents."""
        # Implementation here
        pass
```

#### Configuration Management
```python
# Use Pydantic for configuration validation
from pydantic import BaseModel, Field, validator

class RetrieverConfig(BaseModel):
    """Configuration for document retriever."""
    
    type: str = Field(..., description="Retriever type")
    top_k: int = Field(5, ge=1, le=100, description="Number of results")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ['similarity', 'bm25', 'hybrid']:
            raise ValueError(f'Invalid retriever type: {v}')
        return v
```

## Development Tools

### Pre-commit Hooks

Install and configure pre-commit:
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### Code Formatting

#### Black Configuration
Create `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["rag_engine"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Testing

#### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── fixtures/                # Test data
│   ├── documents/
│   └── configs/
├── unit/                    # Unit tests
│   ├── test_loaders.py
│   ├── test_chunkers.py
│   ├── test_embedders.py
│   ├── test_retrievers.py
│   └── test_orchestration.py
└── integration/             # Integration tests
    ├── test_api.py
    ├── test_pipeline.py
    └── test_end_to_end.py
```

#### Test Configuration
Create `pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --cov=rag_engine
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

#### Example Tests
```python
# tests/unit/test_chunkers.py
import pytest
from rag_engine.core.chunker import FixedSizeChunker

class TestFixedSizeChunker:
    """Test suite for FixedSizeChunker."""
    
    @pytest.fixture
    def chunker_config(self):
        return {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "separator": "\n"
        }
    
    @pytest.fixture
    def chunker(self, chunker_config):
        return FixedSizeChunker(chunker_config)
    
    def test_chunk_text(self, chunker):
        text = "This is a test. " * 20  # Long text
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_chunk_overlap(self, chunker):
        text = "Word " * 50
        chunks = chunker.chunk(text)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap = self._calculate_overlap(chunks[i], chunks[i + 1])
            assert overlap > 0
    
    def _calculate_overlap(self, chunk1: str, chunk2: str) -> int:
        # Calculate character overlap between chunks
        pass

# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from rag_engine.interfaces.api import FastAPIServer

@pytest.fixture
def api_client():
    config = load_test_config()
    server = FastAPIServer(config=config)
    app = server.create_app()
    return TestClient(app)

def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint(api_client):
    response = api_client.post("/chat", json={
        "query": "What is AI?",
        "session_id": "test"
    })
    assert response.status_code == 200
    assert "response" in response.json()
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_chunkers.py

# Run with coverage
pytest --cov=rag_engine --cov-report=html

# Run only unit tests
pytest -m unit

# Run tests in parallel
pytest -n auto
```

## Component Development

### Creating New Components

#### Base Component Structure
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseNewComponent(ABC):
    """Base class for new component type."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate component configuration."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return {
            "name": self.name,
            "type": "new_component",
            "config": self.config
        }
```

#### Implementation Example
```python
from rag_engine.core.base import BaseNewComponent

class CustomNewComponent(BaseNewComponent):
    """Custom implementation of new component."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.parameter = config.get("parameter", "default")
    
    def process(self, input_data: Any) -> Any:
        """Process input data with custom logic."""
        # Implementation here
        return processed_data
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required_keys = ["parameter"]
        return all(key in self.config for key in required_keys)

# Register component
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("new_component", "custom", CustomNewComponent)
```

### Plugin Development

#### Plugin Structure
```python
# plugins/my_plugin.py
from rag_engine.core.base import BaseRetriever
from rag_engine.core.component_registry import ComponentRegistry

class SpecializedRetriever(BaseRetriever):
    """Specialized retriever for specific domain."""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Specialized retrieval logic
        pass

def register_plugin():
    """Register plugin components."""
    ComponentRegistry.register_component(
        "retriever", 
        "specialized", 
        SpecializedRetriever
    )

# Plugin metadata
PLUGIN_INFO = {
    "name": "my_plugin",
    "version": "1.0.0",
    "description": "Specialized retriever plugin",
    "author": "Your Name",
    "components": ["retriever"]
}
```

## Documentation

### Docstring Standards
```python
def complex_function(
    param1: str, 
    param2: int, 
    param3: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Brief description of the function.
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Optional parameter description
        
    Returns:
        Dictionary containing the results with keys:
        - key1: Description of key1
        - key2: Description of key2
        
    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["key1"])
        "value1"
        
    Note:
        This function requires special permissions.
    """
    pass
```

### API Documentation
```python
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    query: str = Field(..., description="User query", example="What is AI?")
    session_id: Optional[str] = Field(None, description="Session identifier")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    model: str = Query("gpt-3.5-turbo", description="LLM model to use")
):
    """
    Chat with the RAG system.
    
    This endpoint processes user queries using the configured RAG pipeline
    and returns intelligent responses based on the document knowledge base.
    
    - **query**: The user's question or prompt
    - **session_id**: Optional session identifier for conversation tracking
    - **model**: LLM model to use for response generation
    """
    pass
```

## Performance Profiling

### Memory Profiling
```python
# Install memory profiler
pip install memory-profiler

# Profile memory usage
@profile
def memory_intensive_function():
    # Your code here
    pass

# Run profiler
python -m memory_profiler script.py
```

### Performance Testing
```python
import time
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile function execution."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    return result
```

## Debugging

### Debug Configuration
```python
import logging

# Development logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Debug Tools
```python
# Debug decorator
def debug_calls(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Breakpoint helper
def debug_breakpoint(locals_dict=None):
    """Debug breakpoint with local variables."""
    import pdb
    if locals_dict:
        for name, value in locals_dict.items():
            print(f"{name}: {value}")
    pdb.set_trace()
```

## Contribution Workflow

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-retriever

# Make changes and commit
git add .
git commit -m "feat: add specialized retriever for domain queries"

# Push and create PR
git push origin feature/new-retriever
```

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```
