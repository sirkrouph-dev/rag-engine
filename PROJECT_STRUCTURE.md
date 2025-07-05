# Project Structure

> **⚠️ EXPERIMENTAL PROJECT ⚠️**
> 
> **This project is in active development. Structure may change as the project evolves.**

```
rag_engine/
├── 📁 archive/                 # Legacy files and completed plans
│   ├── debug_base_api.py       # Legacy debugging script
│   ├── fix_cli.py              # Temporary fix script (completed)
│   ├── RESTRUCTURING_PLAN.md   # Completed restructuring plan
│   └── ...                     # Other archived files
│
├── 📁 configs/                 # Configuration files
│   ├── config.json             # Main configuration file
│   ├── production/             # Production configurations
│   ├── enhanced_production.json # Enhanced production config
│   └── ...                     # Other config files
│
├── 📁 docker/                  # Docker configurations
│   ├── docker-compose.yml      # Development Docker setup
│   ├── docker-compose.demo.yml # Demo Docker setup
│   ├── docker-compose.production.yml # Production Docker setup
│   ├── Dockerfile              # Container definition
│   └── nginx.conf              # Nginx configuration
│
├── 📁 docs/                    # Documentation
│   ├── api/                    # API framework documentation
│   ├── components/             # Component documentation
│   │   ├── chunkers.md
│   │   ├── conversational_routing.md      # Advanced conversational routing system
│   │   ├── conversational_routing_ui.md   # UI integration guide
│   │   ├── embedders.md
│   │   ├── llms.md
│   │   ├── loaders.md
│   │   ├── prompters.md
│   │   ├── retrievers.md
│   │   └── vectorstores.md
│   ├── deployment/             # Deployment guides
│   │   ├── DEPLOYMENT.md       # Main deployment guide
│   │   ├── docker.md           # Docker deployment
│   │   ├── production.md       # Production deployment
│   │   └── SCALING.md          # Scaling guide
│   ├── development/            # Development docs
│   │   ├── architecture.md     # System architecture
│   │   ├── contributing.md     # Contribution guidelines
│   │   └── rag_engine_design.md # Design document
│   └── guides/                 # User guides
│       ├── AI_ASSISTANT_INTEGRATION.md # AI assistant guide
│       ├── BLOAT_REDUCTION.md  # Dependency management
│       ├── DEMO_README.md      # Demo documentation
│       ├── ENHANCED_API_GUIDE.md # Enhanced API guide
│       ├── FRIENDS_DEMO.md     # Friends demo guide
│       ├── GETTING_STARTED.md  # Getting started guide
│       ├── INSTANT_DEMO.md     # Instant demo guide
│       ├── ORCHESTRATION_GUIDE.md # Orchestration guide
│       ├── QUICK_DEMO_SETUP.md # Quick demo setup
│       └── TESTING_GUIDE.md    # Testing guide
│
├── 📁 examples/                # Example configurations and scripts
│   ├── configs/                # Example configuration files
│   ├── scripts/                # Example Python scripts
│   ├── ai_assistant_demo.md    # AI assistant demo example
│   ├── demo_document.md        # Demo document for testing
│   ├── quickstart.md           # Quick start guide
│   └── README.md               # Examples documentation
│
├── 📁 frontend/                # Vue.js frontend application
│   ├── src/                    # Frontend source code
│   │   ├── components/         # Vue components
│   │   │   ├── routing/        # Conversational routing components
│   │   │   │   ├── RoutingConfig.vue      # Routing configuration interface
│   │   │   │   ├── TemplateManager.vue    # Template editing and management
│   │   │   │   ├── RoutingTester.vue      # Query routing testing interface
│   │   │   │   └── RoutingAnalytics.vue   # Analytics and monitoring dashboard
│   │   │   └── ...             # Other components
│   │   ├── views/              # Vue views/pages
│   │   │   ├── AIAssistant.vue # AI assistant interface
│   │   │   ├── Chat.vue        # Chat interface
│   │   │   ├── Dashboard.vue   # Main dashboard
│   │   │   ├── Routing.vue     # Conversational routing management
│   │   │   └── ...             # Other views
│   │   ├── services/           # API services
│   │   └── App.vue             # Main app component
│   ├── package.json            # Frontend dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── FRONTEND_GUIDE.md       # Frontend development guide
│   └── README.md               # Frontend documentation
│
├── 📁 rag_engine/              # Main package
│   ├── config/                 # Configuration modules
│   ├── core/                   # Core RAG components
│   │   ├── component_registry.py # Component registry
│   │   ├── orchestration.py    # Orchestration strategies
│   │   ├── embedder.py         # Embedding components
│   │   └── llm.py              # LLM components
│   ├── interfaces/             # API and CLI interfaces
│   │   ├── api.py              # Base API interface
│   │   ├── cli.py              # Command-line interface
│   │   └── fastapi_enhanced.py # Enhanced FastAPI server
│   └── plugins/                # Plugin system
│
├── 📁 requirements/            # Dependency management
│   ├── base.txt                # Core dependencies
│   └── stacks/                 # Stack-specific requirements
│       ├── demo.txt            # DEMO stack dependencies
│       ├── local.txt           # LOCAL stack dependencies
│       ├── cloud.txt           # CLOUD stack dependencies
│       └── ...                 # Other stack requirements
│
├── 📁 scripts/                 # Utility scripts
│   └── ai_setup.py             # AI-powered setup assistant (only essential script)
│
├── 📁 tests/                   # All test files
│   ├── configs/                # Test configurations
│   ├── fixtures/               # Test data
│   ├── integration/            # Integration tests
│   ├── archived/               # Archived development test files
│   ├── unit/                   # Unit tests
│   ├── test_comprehensive.py   # Comprehensive test suite
│   └── README.md               # Testing documentation
│
├── 📁 .github/                 # GitHub workflows and templates
├── 📁 .vscode/                 # VS Code configuration
├── 📄 .gitignore               # Git ignore rules
├── 📄 instant_demo.bat         # One-click demo setup
├── 📄 PROJECT_STRUCTURE.md     # This file
├── 📄 pyproject.toml           # Python project configuration
├── 📄 pytest.ini              # Test configuration
├── 📄 requirements.txt         # Main Python dependencies
└── 📄 README.md                # Main project documentation
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
│   ├── conversational_routing.md      # Advanced conversational routing system
│   ├── conversational_routing_ui.md   # UI integration guide
│   ├── embedders.md
│   ├── llms.md
│   ├── loaders.md
│   ├── prompters.md
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
├── archived/                   # Archived development test files
├── test_comprehensive.py       # Main comprehensive test suite
└── README.md                   # Testing documentation
```

### `/examples/` - Examples and Demos
```
examples/
├── configs/                    # Example configurations
│   ├── demo_local_config.json  # Local demo setup
│   ├── demo_cloud_config.json  # Cloud demo setup
│   ├── example_config.json     # Basic example
│   └── vertex_ai_example.json  # Google Cloud setup
├── scripts/                    # Example scripts
├── ai_assistant_demo.md        # AI assistant demo
├── demo_document.md            # Demo document for testing
├── quickstart.md               # Quick start guide
└── README.md                   # Examples documentation
```

### `/scripts/` - Essential Utility Scripts
```
scripts/
└── ai_setup.py                 # AI-powered setup assistant
```

**Note**: After cleanup, only the essential AI setup script remains. Redundant batch scripts have been moved to archive.

### `/frontend/` - Vue.js Frontend
```
frontend/
├── src/                        # Frontend source code
│   ├── components/             # Reusable Vue components
│   ├── views/                  # Page components
│   ├── stores/                 # State management (Pinia)
│   ├── services/               # API integration layer
│   └── style.css               # Global styles (Tailwind CSS)
├── package.json                # Frontend dependencies
├── vite.config.js              # Build configuration
├── tailwind.config.js          # Styling configuration
└── README.md                   # Frontend documentation
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
