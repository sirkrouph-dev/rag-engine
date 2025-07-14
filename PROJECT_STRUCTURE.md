# Project Structure

> **🔬 EXPERIMENTAL RAG FRAMEWORK 🔬**
> 
> **Modular RAG engine with conversational AI capabilities. Currently in a highly experimental state. Core functionality is stable, but advanced features are incomplete and under active development.**

```
rag_engine/
├── 📁 archive/                 # Legacy files and completed plans
├── 📁 configs/                 # Configuration files
├── 📁 docker/                  # Docker configurations
├── 📁 docs/                    # Documentation
├── 📁 examples/                # Example configurations and scripts
├── 📁 frontend/                # Vue.js frontend application
├── 📁 rag_engine/              # Main package
│   ├── config/                 # Configuration modules
│   ├── core/                   # Core RAG components (Stable)
│   ├── interfaces/             # API and CLI interfaces (Experimental)
│   └── plugins/                # Plugin system (Future)
├── 📁 requirements/            # Dependency management
├── 📁 scripts/                 # Utility scripts
└── 📁 tests/                   # All test files
```

## 🚧 Current Development Status

### ✅ **Stable Features**
- **Core RAG Pipeline**: `chunker`, `embedder`, `llm`, `loader`, `orchestration`, `retriever`, `vectorstore`.
- **Basic Configuration**: Loading configurations from JSON/YAML is stable.
- **Basic Unit Tests**: Core components have a baseline of passing tests.

### 🔬 **Experimental & Incomplete Features**
- **Advanced RAG Strategies**: Conversational routing and advanced prompting are highly experimental and have significant bugs.
- **API Interfaces**: The `fastapi_enhanced.py` and other API servers are **NOT** production-ready. The code exists, but it is not correctly integrated.
- **Security & Reliability**: All security and reliability features (`security.py`, `monitoring.py`, `reliability.py`) are **stubs or incomplete**. They should not be used.
- **Test Suite**: The majority of the test suite (**~94 tests**) is failing due to the incomplete features above. These tests represent a roadmap for future development, not a confirmation of existing functionality.

**Conclusion**: The project provides a solid foundation for a RAG engine, but the "enterprise" features are a work-in-progress and should be treated as highly experimental. Do not use in a production environment.
