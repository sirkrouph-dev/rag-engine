# Project Structure

> **⚠️ EXPERIMENTAL PROJECT ⚠️**
> 
> **This project is in active development. Structure may change as the project evolves.**

```
rag_engine/
├── 📁 docs/                    # Documentation
│   ├── api/                    # API framework documentation
│   ├── components/             # Component documentation
│   ├── deployment/             # Deployment guides
│   └── development/            # Development docs
│
├── 📁 examples/                # Example configurations and scripts
│   ├── configs/                # Example configuration files
│   ├── scripts/                # Example Python scripts
│   ├── quickstart.md           # Quick start guide
│   └── README.md               # Examples documentation
│
├── 📁 rag_engine/              # Main package
│   ├── config/                 # Configuration modules
│   ├── core/                   # Core RAG components
│   ├── interfaces/             # API and CLI interfaces
│   └── plugins/                # Plugin system
│
├── 📁 tests/                   # All test files
│   ├── configs/                # Test configurations
│   ├── fixtures/               # Test data
│   ├── integration/            # Integration tests
│   ├── unit/                   # Unit tests
│   └── *.py                    # Test modules
│
├── 📁 .github/                 # GitHub workflows and templates
├── 📁 .vscode/                 # VS Code configuration
├── 📄 .gitignore               # Git ignore rules
├── 📄 docker-compose.yml       # Development Docker setup
├── 📄 docker-compose.production.yml  # Advanced Docker setup
├── 📄 Dockerfile               # Container definition
├── 📄 nginx.conf               # Nginx configuration
├── 📄 pyproject.toml           # Python project configuration
├── 📄 pytest.ini              # Test configuration
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # Main project documentation
├── 📄 DEPLOYMENT.md            # Deployment instructions
├── 📄 ENHANCED_API_GUIDE.md    # API usage guide
├── 📄 SCALING.md               # Scaling information
└── 📄 rag_engine_design.md     # Design document
```

## 📚 Core Components

### `/rag_engine/` - Main Package
```
rag_engine/
├── __main__.py                 # CLI entry point
├── config/                     # Configuration system
│   ├── __init__.py
│   ├── loader.py               # Config loading logic
│   └── schema.py               # Config validation schemas
│
├── core/                       # Core RAG functionality
│   ├── __init__.py
│   ├── base.py                 # Base interfaces
│   ├── chunker.py              # Text chunking
│   ├── embedder.py             # Text embedding
│   ├── llm.py                  # Language model integration
│   ├── loader.py               # Document loading
│   ├── orchestration.py        # RAG orchestration
│   ├── pipeline.py             # Pipeline management
│   ├── registry.py             # Component registry
│   ├── retriever.py            # Document retrieval
│   └── vectorstore.py          # Vector storage
│
├── interfaces/                 # External interfaces
│   ├── __init__.py
│   ├── api.py                  # FastAPI interface
│   ├── cli.py                  # Command-line interface
│   ├── flask_api.py            # Flask interface
│   └── ...                     # Other API frameworks
│
└── plugins/                    # Plugin system (future)
    └── __init__.py
```

### `/docs/` - Documentation
```
docs/
├── api/                        # API framework docs
├── components/                 # Component documentation
│   ├── chunkers.md
│   ├── embedders.md
│   ├── llms.md
│   ├── loaders.md
│   ├── retrievers.md
│   └── vectorstores.md
├── deployment/                 # Deployment guides
└── development/                # Development docs
```

### `/tests/` - Test Suite
```
tests/
├── conftest.py                 # Pytest configuration
├── configs/                    # Test configurations
├── fixtures/                   # Test data
├── integration/                # End-to-end tests
├── unit/                       # Component unit tests
└── test_*.py                   # Individual test modules
```

### `/examples/` - Examples and Demos
```
examples/
├── configs/                    # Example configurations
│   ├── basic.json             # Simple setup
│   ├── vertex_ai.json         # Google Cloud setup
│   └── hybrid.json            # Advanced features
├── scripts/                    # Example scripts
└── quickstart.md               # Getting started guide
```

## 🔧 Development Workflow

1. **Core Logic**: Implement in `/rag_engine/core/`
2. **Interfaces**: Add APIs in `/rag_engine/interfaces/`
3. **Configuration**: Define schemas in `/rag_engine/config/`
4. **Testing**: Add tests in `/tests/`
5. **Documentation**: Update docs in `/docs/`
6. **Examples**: Provide examples in `/examples/`

## 📦 Build Artifacts

The following directories are generated during development:
- `.pytest_cache/` - Pytest cache
- `__pycache__/` - Python bytecode cache
- `.venv/` - Virtual environment
- `test_vector_store*/` - Test data (auto-cleaned)

## 🚀 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Test**: `python -m pytest tests/ -v`
3. **Run**: `python -m rag_engine serve --config examples/configs/basic.json`
4. **Develop**: See `/docs/development/contributing.md`

## ⚠️ Notes

- Project structure may evolve during experimental phase
- Some directories are created automatically during development
- See individual README files in each directory for more details
