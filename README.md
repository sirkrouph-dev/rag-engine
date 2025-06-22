# RAG Engine

> **⚠️ EXPERIMENTAL - WORK IN PROGRESS ⚠️**
> 
> **This framework is currently in active development and experimental phase. Features, APIs, and configurations may change significantly. Not recommended for production use yet.**

A **modular framework** for building advanced Retrieval-Augmented Generation (RAG) pipelines with **multi-framework API support**, **modular orchestration**, and **comprehensive deployment capabilities**.

## ✨ **Key Features**

### 🏗️ **Modular Architecture**
- **Component Registry**: Swap retrievers, LLMs, embedders without code changes
- **Orchestration Layer**: Default, hybrid, and multi-modal RAG strategies
- **Configuration-Driven**: Control everything via JSON/YAML configs

### � **Multi-Framework APIs**
- **FastAPI**: High-performance async API with auto-generated docs
- **Flask**: Lightweight web framework with CORS support
- **Django REST**: Enterprise framework support
- **Custom Servers**: Add your own server implementations

### 🎨 **Modern Frontend Interface**
- **Vue.js Application**: Beautiful, responsive web interface
- **Interactive Chat**: Real-time chat with document sources
- **Dashboard**: System monitoring and pipeline management
- **Document Browser**: Explore documents and chunks
- **System Monitor**: Comprehensive health and status display

### 🐳 **Deployment Capabilities**
- **Docker**: Multi-stage builds with deployment configs
- **Scaling**: Multi-worker support with load balancing
- **Monitoring**: Health checks, metrics, and logging
- **Security**: Authentication, CORS, rate limiting

## 🚀 **Quick Start**

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server with example config
python -m rag_engine serve --config examples/configs/example_config.json --port 8000
```

### Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Access the web interface at `http://localhost:3000` and the API at `http://localhost:8000/docs`.

## 📚 **Documentation**

### **Project Organization**
- [**Project Structure**](PROJECT_STRUCTURE.md) - Complete directory organization
- [**Examples**](examples/) - Configuration examples and sample scripts

### **Core Concepts**
- [**Orchestration Guide**](docs/orchestration.md) - Modular orchestration layer
- [**Configuration**](docs/configuration.md) - Config schemas and examples
- [**Components**](docs/components/) - Individual component documentation

### **API Frameworks**
- [**FastAPI**](docs/api/fastapi.md) - FastAPI implementation and features
- [**Custom Servers**](docs/api/custom-servers.md) - Creating custom server implementations

### **Frontend Interface**
- [**Vue.js Frontend**](frontend/README.md) - Modern web interface documentation
- [**Frontend Guide**](frontend/FRONTEND_GUIDE.md) - Complete frontend development guide
- [**UI/UX Features**](frontend/FRONTEND_GUIDE.md#uiux-design-principles) - Design principles and user experience

### **Deployment**
- [**Docker**](docs/deployment/docker.md) - Containerized deployment
- [**Development Deployment**](docs/deployment/production.md) - Development configuration
- [**Scaling**](docs/deployment/scaling.md) - Multi-worker and load balancing
- [**Monitoring**](docs/deployment/monitoring.md) - Health checks and metrics

### **Development**
- [**Contributing**](docs/development/contributing.md) - Development guidelines
- [**Testing**](docs/development/testing.md) - Test suite and coverage
- [**Architecture**](docs/development/architecture.md) - System architecture

## 🎯 **Use Cases**

- **Document Q&A**: Query your documents with natural language
- **Knowledge Bases**: Build searchable knowledge repositories
- **Research Tools**: Advanced retrieval for research workflows
- **Customer Support**: AI-powered support with document grounding
- **Content Analysis**: Extract insights from large document collections

## 📦 **Components**

| Component | Options | Description |
|-----------|---------|-------------|
| **Loaders** | txt, pdf, docx, html | Document loading and parsing |
| **Chunkers** | fixed_size, sentence, token | Text segmentation strategies |
| **Embedders** | huggingface, openai, gemini/vertex-ai, local | Text embedding models |
| **Vector Stores** | chroma, faiss, pinecone | Vector database backends |
| **Retrievers** | similarity, bm25, hybrid, mmr | Document retrieval methods |
| **LLMs** | openai, anthropic, local, ollama | Language model providers |
| **Orchestrators** | default, hybrid, multimodal | RAG pipeline strategies |

## � **Example Configurations**

### **Basic Setup**
```json
{
  "documents": [{"path": "docs/", "type": "directory"}],
  "chunking": {"method": "fixed_size", "chunk_size": 500},
  "embedding": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
  "vectorstore": {"provider": "chroma", "persist_directory": "./db"},
  "retrieval": {"method": "similarity", "top_k": 5},
  "llm": {"provider": "openai", "model": "gpt-4"}
}
```

### **Hybrid Retrieval**
```json
{
  "retrieval": {
    "method": "hybrid",
    "semantic_weight": 0.7,
    "bm25_weight": 0.3,
    "top_k": 10
  }
}
```

### **Google Vertex AI**
```json
{
  "embedding": {
    "type": "gemini",
    "use_vertex": true,
    "model": "textembedding-gecko@001",
    "project": "your-gcp-project",
    "location": "us-central1"
  }
}
```

## 🚀 **CLI Commands**

```bash
# Initialize new project
python -m rag_engine init [project-name]

# Build vector database
python -m rag_engine build --config config.json

# Interactive chat
python -m rag_engine chat --config config.json

# Serve API
python -m rag_engine serve --config config.json [options]

# Custom server management
python -m rag_engine custom-server list
python -m rag_engine custom-server create --name myserver
```

## 🐳 **Docker Deployment**

```bash
# Build and run
docker-compose up --build

# Development deployment (experimental)
docker-compose -f docker-compose.production.yml up -d

# Scale workers
docker-compose up --scale rag-engine=3
```

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=rag_engine

# Test specific component
python -m pytest tests/unit/test_chunker.py -v
```

## 📊 **Status**

- ✅ **59/59 Tests Passing**
- ⚠️ **Experimental Phase**
- ✅ **Docker Support**
- ✅ **Multi-Framework APIs**
- ✅ **Modular Orchestration**
- ✅ **Custom Server Support**
- ✅ **Comprehensive Documentation**
- ⚠️ **NOT Ready for Production**

## 🤝 **Contributing**

See [Contributing Guide](docs/development/contributing.md) for development setup, coding standards, and contribution guidelines.

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ and GitHub Copilot** - Professional-grade RAG infrastructure for modern applications.
curl http://localhost:8000/health

# Build pipeline  
curl -X POST http://localhost:8000/build

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?"}'
```

### **Docker Deployment**
```bash
# Single container
docker build -t rag-engine .
docker run -p 8000:8000 rag-engine

# Full stack with load balancer
docker-compose up -d
```

## 🗺️ **Implementation Status**

### ✅ **Implemented Components**

| Component | Status | Key Features |
|-----------|--------|--------------|
| **🏗️ Multi-Framework APIs** | ✅ **Functional** | FastAPI, Flask, Django REST with seamless switching |
| **🎨 Web UIs** | ✅ **Functional** | Streamlit analytics + Gradio chat interfaces |
| **⚙️ Core Pipeline** | ✅ **Functional** | Modular, tested, configurable |
| **🖥️ CLI Interface** | ✅ **Functional** | Full command suite (init, build, chat, serve) |
| **🔧 Configuration** | ✅ **Functional** | Pydantic validation, env vars, YAML/JSON |
| **📚 Document Loading** | ✅ **Functional** | TXT, PDF, DOCX, HTML with error handling |
| **🧩 Text Chunking** | ✅ **Functional** | Token-based, recursive, configurable overlap |
| **🔢 Embeddings** | ✅ **Functional** | HuggingFace, OpenAI, Vertex AI with batch processing |
| **💾 Vector Storage** | ✅ **Functional** | ChromaDB with persistence and querying |
| **🤖 LLM Integration** | ✅ **Functional** | OpenAI, HuggingFace, local models with provider factory |
| **🔍 Retrieval System** | ✅ **Functional** | Similarity search with configurable top-k |
| **🧪 Testing Suite** | ✅ **Functional** | 59 tests passing, unit + integration coverage |
| **🐳 Docker Deployment** | ✅ **Functional** | Multi-container setup with Nginx load balancing |
| **📊 Health Monitoring** | ✅ **Functional** | Health checks, status endpoints, error handling |

### 🟡 **Advanced Features (Ready for Extension)**

| Component | Status | Implementation Details |
|-----------|--------|----------------------|
| **🔧 Tools & Plugins** | 🟡 **Framework Ready** | Web search, calculator, file operations framework |
| **🧠 Reasoning Engine** | 🟡 **Framework Ready** | Chain-of-thought, tree-of-thought patterns |
| **🔐 Authentication** | 🟡 **Configurable** | API key auth, rate limiting framework |
| **📈 Analytics** | 🟡 **Basic** | Request logging, basic metrics |

### 🔴 **Future Enhancements**

| Component | Status | Roadmap |
|-----------|--------|---------|
| **🔧 Plugin Marketplace** | 🔴 **Planned** | Extensible plugin system for custom components |
| **🎯 Fine-tuning Integration** | 🔴 **Planned** | Model customization and training workflows |
| **☁️ Cloud Deployment** | 🔴 **Planned** | AWS, GCP, Azure deployment templates |
| **📊 Advanced Analytics** | 🔴 **Planned** | Performance dashboards, usage analytics |

## 🏗️ **API Framework Examples**

### **FastAPI (Recommended for Development)**
```bash
# Start FastAPI server
python -m rag_engine serve --config config.json --framework fastapi --port 8000

# Features:
# ✅ Automatic OpenAPI docs at /docs
# ✅ High-performance async operations  
# ✅ Type validation with Pydantic
# ✅ Built-in health checks
```

### **Flask (Lightweight Alternative)**
```bash
# Start Flask server  
python -m rag_engine serve --config config.json --framework flask --port 8001

# Features:
# ✅ Simple and flexible
# ✅ CORS enabled
# ✅ Easy debugging
# ✅ Quick prototyping
```

### **Streamlit UI (Interactive Dashboard)**
```bash
# Start with Streamlit interface
python -m rag_engine serve --config config.json --ui streamlit --ui-port 8501

# Features:
# ✅ Chat interface
# ✅ Configuration editor
# ✅ Analytics dashboard
# ✅ Document upload
```

## 🖥️ **CLI Commands**

### **Project Management**
```bash
# Initialize new project
python -m rag_engine init my-project --template advanced

# Build pipeline from config
python -m rag_engine build --config config.json

# Interactive chat session
python -m rag_engine chat --config config.json
```

### **Server Management**  
```bash
# Start API server (choose framework)
python -m rag_engine serve --framework fastapi --port 8000
python -m rag_engine serve --framework flask --port 8001

# Start with UI
python -m rag_engine serve --ui streamlit --ui-port 8501
python -m rag_engine serve --ui gradio --ui-port 7860

# Development mode with auto-reload
python -m rag_engine serve --reload --framework fastapi
```

## � **API Documentation**

### **Core Endpoints**
| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/health` | GET | Health check | `{"status": "healthy"}` |
| `/status` | GET | System status | `{"pipeline_built": true, "config": {...}}` |
| `/build` | POST | Build/rebuild pipeline | `{"status": "success", "documents": 10}` |
| `/chat` | POST | Interactive chat | `{"query": "...", "response": "..."}` |
| `/config` | GET | Configuration info | `{"embedding_provider": "openai"}` |
| `/documents` | GET | List loaded docs | `{"documents": [...], "total": 5}` |
| `/chunks` | GET | List document chunks | `{"chunks": [...], "total": 50}` |

### **Request/Response Examples**

#### **Chat Request**
```json
{
  "query": "What are the main features?",
  "session_id": "user123"  
}
```

#### **Chat Response** 
```json
{
  "query": "What are the main features?",
  "response": "The main features include...",
  "session_id": "user123",
  "status": "success"
}
```

## 🧪 **Testing & Quality**

### **Test Coverage**
- ✅ **59 tests passing** (100% success rate)
- ✅ **Unit tests**: Core components (chunker, embedder, vectorstore)
- ✅ **Integration tests**: CLI, pipeline, API endpoints
- ✅ **Zero warnings**: Proper pytest configuration

### **Run Tests**
```bash
# All tests
python -m pytest tests/ -v

# Specific categories
python -m pytest tests/unit/ -v       # Unit tests only
python -m pytest tests/integration/ -v # Integration tests only

# With coverage
python -m pytest --cov=rag_engine tests/
```

## 🐳 **Development Deployment**

### **Docker Single Container**
```bash
# Build and run
docker build -t rag-engine .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" rag-engine
```

### **Docker Compose (Multi-Service)**
```bash
# Start full stack
docker-compose up -d

# Services included:
# - FastAPI server (port 8000)
# - Flask server (port 8001)  
# - Streamlit UI (port 8501)
# - Nginx load balancer (port 80)
```

### **Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACE_API_TOKEN="your-hf-token"
export RAG_CONFIG_PATH="/path/to/config.json"
```

## 📖 **Configuration Examples**

### **Simple Configuration**
```json
{
  "documents": [
    {"type": "txt", "path": "./documents/"}
  ],
  "chunking": {
    "method": "fixed", 
    "max_tokens": 200, 
    "overlap": 20
  },
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "./vector_store"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo"
  }
}
```

### **Production Configuration**

⚠️ **Note: This is an experimental configuration example. Not recommended for actual production use.**

```json
{
  "documents": [
    {
      "type": "directory",
      "path": "/app/data/documents",
      "recursive": true,
      "file_types": ["txt", "pdf", "md", "docx"]
    }
  ],
  "chunking": {
    "method": "recursive",
    "max_tokens": 512,
    "overlap": 50
  },
  "embedding": {
    "provider": "openai", 
    "model": "text-embedding-3-small",
    "api_key": "${OPENAI_API_KEY}"
  },
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "/app/vector_store",
    "collection_name": "production_docs"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4-turbo-preview", 
    "temperature": 0.1
  },
  "security": {
    "enable_auth": true,
    "rate_limit": {
      "requests_per_minute": 60
    }
  },
  "performance": {
    "cache_enabled": true,
    "async_processing": true
  }
}
```

## 🚀 **What's New in This Release**

### **🎉 Major Features Added**
- ✅ **Multi-Framework API Support**: FastAPI, Flask, Django REST
- ✅ **Integrated Web UIs**: Streamlit analytics + Gradio chat
- ✅ **Docker Deployment**: Docker + Compose + Nginx
- ✅ **Enterprise Features**: Health monitoring, error handling, CORS
- ✅ **Comprehensive Testing**: 59 tests, zero warnings
- ✅ **Professional CLI**: Full command suite with help system
- ✅ **Vertex AI Support**: Google Cloud embeddings with service account auth

### **🔧 Technical Improvements**
- ✅ **Lazy Loading**: Fast startup with on-demand pipeline initialization  
- ✅ **Factory Pattern**: Extensible framework registration system
- ✅ **Error Handling**: Graceful degradation throughout the stack
- ✅ **Configuration**: Environment variables with Pydantic validation
- ✅ **Documentation**: Auto-generated API docs with examples

## ⚠️ **Development Status**

This RAG Engine is currently in **active development and experimental phase**:

### **🧪 For Research & Development**
- **Flexible** framework switching for experimentation
- **Extensible** plugin architecture
- **Interactive** UIs for rapid prototyping
- **Comprehensive** testing for reliable development

### **🏗️ Architecture Ready**
- **Containerized** with Docker
- **Load balanced** with Nginx
- **Environment-aware** configuration
- **Health monitored** for reliability

### **⚠️ Not Yet Ready For**
- **Production** workloads
- **Enterprise** deployments
- **Mission-critical** applications
- **Long-term** stability guarantees

## 📞 **Getting Help**

### **Documentation**
- 📖 **API Docs**: Available at `/docs` when running FastAPI
- 📋 **Deployment Guide**: See `DEPLOYMENT.md` for detailed instructions
- 🧪 **Testing Guide**: Run `python -m pytest tests/ -v`

### **Community & Support**
- 🐛 **Issues**: Report bugs and request features on GitHub
- 💡 **Discussions**: Join our community discussions
- 🤝 **Contributing**: See contribution guidelines for developers

---

**The RAG Engine: Experimental Framework for Advanced RAG Pipelines!** 🧪

Built with modern architecture principles, comprehensive testing, and flexible deployment capabilities. Currently in active development for research and experimentation.

_Experiment. Learn. Contribute._ ✨

