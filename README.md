# RAG Engine

> **🔬 EXPERIMENTAL RAG FRAMEWORK 🔬**
> 
> **A modular, AI-powered framework for building advanced Retrieval-Augmented Generation (RAG) pipelines. Currently in active development with significant progress on enterprise features.**

This framework provides a solid foundation for building RAG systems with a focus on modularity and configuration-driven design. The core RAG components are stable, and we've made significant progress on enterprise features including production caching and security integration.

## 🚧 **Project Status: ACTIVE DEVELOPMENT** 🚧

The project has evolved significantly from its initial experimental state. Here's the current status:

### ✅ **Stable & Production-Ready Features**
- **Core RAG Pipeline**: The fundamental components (`loader`, `chunker`, `embedder`, `vectorstore`, `retriever`, `llm`) are functional and can be used to build a basic RAG pipeline.
- **Configuration System**: Loading configurations from JSON/YAML is stable.
- **Conversational Routing**: Advanced conversational AI capabilities with intelligent routing between different RAG strategies.
- **Enhanced Prompting**: Sophisticated prompt engineering with context-aware templates.
- **Production Caching**: **100% test coverage** - Complete caching system with Redis and in-memory providers, rate limiting, and session management.
- **Basic Unit Tests**: Core components have comprehensive test coverage.

### 🔬 **In Development & Partially Complete**
- **Security Integration**: **65% test coverage** - Authentication (JWT, API keys), input validation, audit logging, and rate limiting are functional. Some edge cases and advanced features still need refinement.
- **AI-Powered Setup**: The `ai_setup.py` script is a helpful tool for getting started.
- **Frontend**: A basic Vue.js frontend is available for simple chat demonstrations.

### 🚧 **Still Experimental & Incomplete**
- **Advanced API Interfaces**: The `fastapi_enhanced.py` server and other API implementations need integration work.
- **Monitoring & Reliability**: Advanced monitoring, circuit breakers, and reliability features are still in development.
- **Test Suite**: While we've made significant progress (many tests now passing), some advanced features still have failing tests that serve as a development roadmap.

**Conclusion**: This project has evolved from purely experimental to having solid core functionality with significant enterprise features. While not yet fully production-ready, it's much closer to being enterprise-grade than before.

## ✨ **Key Features (Core & Stable)**

### 🤖 **AI-Powered Setup & Management**
- **Intelligent Setup Assistant**: A local LLM can guide you through basic configuration.
- **Smart Stack Selection**: Presets for `DEMO`, `LOCAL`, `CLOUD` to manage dependencies.

### 🏗️ **Modular Architecture**
- **Component Registry**: Swap core components without changing code.
- **Configuration-Driven**: Control the stable RAG pipeline via JSON/YAML configs.

### 🎨 **Modern Frontend Interface**
- **Vue.js Application**: A simple web interface for demonstrating the chat functionality.

### 🚀 **Enterprise Features (In Progress)**
- **Production Caching**: Redis and in-memory caching with rate limiting and session management
- **Security Framework**: JWT authentication, API key validation, input sanitization, and audit logging
- **Conversational AI**: Advanced routing and context-aware prompting

## 🚀 **Quick Start**

The project is now more stable and ready for experimentation:

### 💻 **Manual Setup (Recommended)**
```bash
# Install base dependencies
pip install -r requirements/base.txt

# Choose your stack (demo is recommended for stability):
pip install -r requirements/stacks/demo.txt

# Start with a basic config (see examples/)
python -m rag_engine serve --config examples/configs/example_config.json
```

### 🎨 **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
```

## 📊 **Recent Progress**

### ✅ **Major Achievements**
- **ProductionCacheManager**: 100% test pass rate (34/34 tests)
- **SecurityIntegration**: 65% test pass rate (15/23 tests)
- **Core RAG Features**: All conversational routing and enhanced prompting tests passing
- **Interface Compatibility**: Fixed major async/sync interface mismatches

### 🎯 **Current Focus**
- Completing security integration test fixes
- Polishing input validation and edge cases
- Finalizing API interface integration

## 📚 **Documentation**

- [**🏗️ Project Structure**](PROJECT_STRUCTURE.md) - Understand the project layout.
- [**🧪 Testing Guide**](docs/guides/TESTING_GUIDE.md) - See how to run the test suite.
- [**🧩 Components**](docs/components/) - Documentation for the stable, core components.

---
*The project has made significant strides toward enterprise-readiness. While some advanced features are still in development, the core functionality is solid and many enterprise features are now functional.*
