# RAG Engine

> **⚠️ EXPERIMENTAL - WORK IN PROGRESS ⚠️**
> 
> **This framework is currently in active development and experimental phase. Features, APIs, and configurations may change significantly. Not recommended for production use yet.**

A **modular, AI-powered framework** for building advanced Retrieval-Augmented Generation (RAG) pipelines with **zero-bloat installation**, **intelligent stack selection**, and **user-friendly automation**.

## ✨ **Key Features**

### 🤖 **AI-Powered Setup & Management**
- **Intelligent Setup Assistant**: Local LLM guides you through configuration
- **Smart Stack Selection**: DEMO, LOCAL, CLOUD, MINI, FULL, RESEARCH presets
- **Bloat Management**: Install only what you need, when you need it
- **Ongoing Help**: Ask AI assistant questions anytime with `rag-engine ask`

### ⚡ **Instant Demo & Quick Start**
- **One-Click Demo**: `instant_demo.bat` for zero-Python setup
- **Docker Compose Demo**: Complete stack in containers
- **Modular Requirements**: Tiered dependencies for lean installs
- **Auto-Detection**: Smart config discovery and validation

### 🏗️ **Modular Architecture**
- **Component Registry**: Swap retrievers, LLMs, embedders without code changes
- **Orchestration Layer**: Default, hybrid, and multi-modal RAG strategies
- **Plugin-Based**: Extensible, lazy-loaded components
- **Configuration-Driven**: Control everything via JSON/YAML configs

### 🎨 **Modern Frontend Interface**
- **Vue.js Application**: Beautiful, responsive web interface with dark mode
- **AI Assistant View**: Interactive chat for stack management and help
- **Interactive Chat**: Real-time chat with document sources
- **Dashboard**: System monitoring and pipeline management
- **Document Browser**: Explore documents and chunks

### 🚀 **Multi-Framework APIs**
- **FastAPI**: High-performance async API with auto-generated docs
- **Enhanced Endpoints**: AI assistant, stack management, orchestrator APIs
- **Custom Servers**: Add your own server implementations
- **Health Monitoring**: Comprehensive status and error tracking

## 🚀 **Quick Start**

### 🎯 **Instant Demo (Recommended)**
```batch
# Windows users - one-click demo setup
.\instant_demo.bat

# This will:
# 1. Check Python/Node.js requirements
# 2. Install demo dependencies only
# 3. Start backend + frontend automatically
# 4. Open browser to demo interface
```

### 🐳 **Docker Demo (Zero Python Setup)**
```bash
# Complete RAG Engine in containers
docker-compose -f docker/docker-compose.demo.yml up

# Access frontend: http://localhost:3000
# Access backend: http://localhost:8000
```

### 🤖 **AI-Powered Setup**
```bash
# Let AI assistant guide your setup
python scripts/ai_setup.py

# The AI will:
# 1. Install a local LLM for guidance
# 2. Analyze your needs and recommend a stack
# 3. Install only required dependencies
# 4. Configure your ideal setup
```

### 💻 **Manual Setup**
```bash
# Install base dependencies
pip install -r requirements/base.txt

# Choose your stack (see BLOAT_REDUCTION.md for details):
pip install -r requirements/stacks/demo.txt      # Minimal demo
pip install -r requirements/stacks/local.txt     # Local models only
pip install -r requirements/stacks/cloud.txt     # Cloud providers
pip install -r requirements/stacks/full.txt      # Everything

# Start with optional config auto-detection
python -m rag_engine serve
```

### 🎨 **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
```

## 🤖 **AI Assistant Commands**

```bash
# Get help anytime
rag-engine ask "How do I configure Vertex AI?"
rag-engine ask "What's the best stack for my use case?"

# AI-powered bloat management
rag-engine analyze-bloat           # Check dependency usage
rag-engine optimize-stack          # Remove unused dependencies
rag-engine dependency-audit        # Security and update audit

# Stack management
rag-engine list-stacks            # Show available preset stacks
rag-engine switch-stack CLOUD     # Switch to cloud stack
rag-engine install-stack LOCAL    # Install local models stack
```

## 📚 **Documentation**

### **Getting Started**
- [**🚀 Instant Demo Guide**](docs/guides/INSTANT_DEMO.md) - Zero-setup demo walkthrough
- [**🏗️ Project Structure**](PROJECT_STRUCTURE.md) - Complete directory organization
- [**🧪 Testing Guide**](docs/guides/TESTING_GUIDE.md) - Validate your setup

### **Core Concepts**
- [**🤖 AI Assistant Integration**](AI_ASSISTANT_INTEGRATION.md) - AI-powered help system
- [**📦 Bloat Reduction**](BLOAT_REDUCTION.md) - Modular installation strategies
- [**⚙️ Orchestration Guide**](docs/orchestration.md) - Modular orchestration layer
- [**📋 Configuration**](docs/configuration.md) - Config schemas and examples

### **Components & APIs**
- [**🧩 Components**](docs/components/) - Individual component documentation
- [**🚀 FastAPI**](docs/api/fastapi.md) - FastAPI implementation and features
- [**🎨 Frontend Guide**](frontend/FRONTEND_GUIDE.md) - Complete frontend development guide

### **Deployment & Development**
- [**🐳 Docker**](docs/deployment/docker.md) - Containerized deployment
- [**🔧 Contributing**](docs/development/contributing.md) - Development guidelines
- [**🏗️ Architecture**](docs/development/architecture.md) - System architecture

## 📦 **Stack Presets**

| Stack | Abbreviation | Use Case | Size | Components |
|-------|-------------|----------|------|------------|
| **Demo** | `DEMO` | Quick demos, testing | ~50MB | HuggingFace embeddings, ChromaDB, OpenAI |
| **Local** | `LOCAL` | Privacy-focused, offline | ~2GB | Local LLMs, offline embeddings |
| **Cloud** | `CLOUD` | Production-ready, scalable | ~100MB | Cloud providers (OpenAI, Vertex AI) |
| **Mini** | `MINI` | Minimal footprint | ~20MB | Essential components only |
| **Full** | `FULL` | Complete feature set | ~5GB | All providers and models |
| **Research** | `RESEARCH` | Advanced features | ~3GB | Experimental components |

## 🧩 **Components**

| Component | Options | Description |
|-----------|---------|-------------|
| **Loaders** | txt, pdf, docx, html | Document loading and parsing |
| **Chunkers** | fixed_size, sentence, token | Text segmentation strategies |
| **Embedders** | huggingface, openai, gemini/vertex-ai, local | Text embedding models |
| **Vector Stores** | chroma, faiss, pinecone | Vector database backends |
| **Retrievers** | similarity, bm25, hybrid, mmr | Document retrieval methods |
| **LLMs** | openai, anthropic, local, ollama | Language model providers |
| **Orchestrators** | default, hybrid, multimodal | RAG pipeline strategies |

## 🌟 **Example Configurations**

### **Demo Configuration**
```json
{
  "documents": [{"path": "demo_document.md", "type": "file"}],
  "chunking": {"method": "fixed_size", "chunk_size": 500},
  "embedding": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
  "vectorstore": {"provider": "chroma", "persist_directory": "./demo_db"},
  "retrieval": {"method": "similarity", "top_k": 5},
  "llm": {"provider": "openai", "model": "gpt-3.5-turbo"}
}
```

### **Local Stack Configuration**
```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "local": true
  },
  "llm": {
    "provider": "ollama",
    "model": "llama2",
    "local": true
  }
}
```

### **Cloud Stack with Vertex AI**
```json
{
  "embedding": {
    "type": "gemini",
    "use_vertex": true,
    "model": "textembedding-gecko@001",
    "project": "your-gcp-project",
    "location": "us-central1"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4-turbo"
  }
}
```

## 🖥️ **CLI Commands**

### **Core Commands**
```bash
# Interactive chat with auto-config detection
python -m rag_engine chat

# Serve API with config auto-detection
python -m rag_engine serve

# Build pipeline from config
python -m rag_engine build --config config.json

# Initialize new project
python -m rag_engine init my-project
```

### **AI Assistant Commands**
```bash
# Get AI help anytime
python -m rag_engine ask "How do I configure embeddings?"

# Stack management
python -m rag_engine list-stacks
python -m rag_engine install-stack DEMO
python -m rag_engine switch-stack LOCAL

# Bloat management
python -m rag_engine analyze-bloat
python -m rag_engine optimize-stack
python -m rag_engine dependency-audit
```

### **Advanced Commands**
```bash
# Orchestrator management
python -m rag_engine list-orchestrators
python -m rag_engine use-orchestrator hybrid

# Component management
python -m rag_engine list-components
python -m rag_engine test-component embedder
```

## 🔗 **API Endpoints**

### **Core RAG APIs**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Interactive chat |
| `/build` | POST | Build/rebuild pipeline |
| `/documents` | GET | List loaded documents |
| `/chunks` | GET | List document chunks |

### **AI Assistant APIs**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ai/ask` | POST | Ask AI assistant questions |
| `/ai/stacks` | GET | List available stacks |
| `/ai/analyze-bloat` | POST | Analyze dependency usage |
| `/ai/optimize-stack` | POST | Optimize current stack |

### **Orchestrator APIs**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestrators` | GET | List available orchestrators |
| `/orchestrators/{name}` | GET | Get orchestrator details |
| `/orchestrators/current` | GET | Get current orchestrator |

## 🧪 **Testing**

### **Quick Validation**
```bash
# Run comprehensive test suite
python test_comprehensive.py

# Test specific components
python -m pytest tests/unit/test_chunking.py -v
python -m pytest tests/integration/test_api_endpoints.py -v

# Test with coverage
python -m pytest --cov=rag_engine tests/
```

### **Demo Validation**
Follow the [Testing Guide](docs/guides/TESTING_GUIDE.md) to validate:
- ✅ Instant demo setup works
- ✅ Docker demo works
- ✅ AI assistant responds
- ✅ Frontend connects to backend
- ✅ Chat interface works

## 🐳 **Docker Deployment**

### **Demo Deployment**
```bash
# Zero-Python setup - everything in containers
docker-compose -f docker/docker-compose.demo.yml up

# Services:
# - Backend API (port 8000)
# - Frontend UI (port 3000)
# - Vector database (ChromaDB)
```

### **Development Deployment**
```bash
# Full development stack
docker-compose -f docker/docker-compose.yml up --build

# Includes:
# - Load balancer (Nginx)
# - Multiple backend workers
# - Development tools
```

## 📊 **Project Status**

### ✅ **Implemented & Tested**
- 🤖 **AI Assistant Integration** - Local LLM setup, ongoing help, stack management
- ⚡ **Instant Demo Setup** - One-click Windows setup, Docker demo
- 📦 **Modular Dependencies** - Tiered requirements, preset stacks
- 🎨 **Modern Frontend** - Vue.js with dark mode, AI assistant view
- 🏗️ **Enhanced APIs** - New endpoints for AI assistant and orchestrators
- 🧪 **Comprehensive Testing** - All major features validated

### 🟡 **Experimental/WIP**
- 🔧 **Plugin System** - Framework ready, needs marketplace
- 📈 **Advanced Analytics** - Basic metrics, needs dashboards
- ☁️ **Cloud Deployment** - Docker ready, needs cloud templates

### ❌ **Not Ready For**
- **Production** workloads
- **Enterprise** deployments  
- **Mission-critical** applications

## 🎯 **Use Cases**

- **📚 Document Q&A**: Query your documents with natural language
- **🧠 Knowledge Bases**: Build searchable knowledge repositories  
- **🔬 Research Tools**: Advanced retrieval for research workflows
- **🎓 Educational Tools**: AI-powered learning and tutoring systems
- **💬 Customer Support**: AI-powered support with document grounding
- **📊 Content Analysis**: Extract insights from large document collections

## 🤝 **Contributing**

### **Getting Started**
1. **Fork & Clone**: `git clone your-fork-url`
2. **AI Setup**: `python scripts/ai_setup.py` (let AI guide you)
3. **Install Dev Stack**: `pip install -r requirements/stacks/full.txt`
4. **Run Tests**: `python test_comprehensive.py`

### **Development Workflow**
- 🔍 **Issues**: Check GitHub issues for contribution opportunities
- 📝 **Pull Requests**: Follow our contribution guidelines
- 🧪 **Testing**: Add tests for new features
- 📚 **Documentation**: Update docs for any changes

See [Contributing Guide](docs/development/contributing.md) for detailed guidelines.

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

**🚀 The RAG Engine: AI-Powered, Zero-Bloat, User-Friendly RAG Framework**

*Built for developers who want intelligent assistance, modular architecture, and instant demos without the complexity overhead.*

**Experiment. Learn. Build. 🧪✨**
