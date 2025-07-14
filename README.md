# RAG Engine

> **🔬 EXPERIMENTAL RAG FRAMEWORK 🔬**
> 
> **A modular, AI-powered framework for building advanced Retrieval-Augmented Generation (RAG) pipelines. Currently in a highly experimental state.**

This framework provides a solid foundation for building RAG systems with a focus on modularity and configuration-driven design. While the core RAG components are stable, many of the advanced, "enterprise" features are currently incomplete stubs and should not be used in production.

## 🚧 **Project Status: EXPERIMENTAL** 🚧

It is important to understand the current state of the project before using it:

### ✅ **Stable Features**
- **Core RAG Pipeline**: The fundamental components (`loader`, `chunker`, `embedder`, `vectorstore`, `retriever`, `llm`) are functional and can be used to build a basic RAG pipeline.
- **Configuration System**: Loading configurations from JSON/YAML is stable.
- **Basic Unit Tests**: The core components have a baseline of passing tests that validate their functionality.
- **AI-Powered Setup**: The `ai_setup.py` script is a helpful tool for getting started.
- **Frontend**: A basic Vue.js frontend is available for simple chat demonstrations.

### 🔬 **Experimental & Incomplete Features**
- **Advanced RAG Strategies**: Conversational routing and advanced prompting are highly experimental. The tests for these are largely failing.
- **API Interfaces**: The `fastapi_enhanced.py` server and other API implementations are **NOT production-ready**. They are stubs for future development.
- **Security & Reliability**: All advanced security and reliability features (Authentication, Rate Limiting, Circuit Breakers, etc.) are **stubs or incomplete**. They are not functional.
- **Test Suite**: The majority of the test suite (**over 90 tests**) is **currently failing**. These failing tests serve as a development roadmap, not as a validation of existing features.

**Conclusion**: This project is an excellent starting point for RAG experimentation but is **NOT enterprise-ready**. Do not use it in a production environment.

## ✨ **Key Features (Core & Stable)**

### 🤖 **AI-Powered Setup & Management**
- **Intelligent Setup Assistant**: A local LLM can guide you through basic configuration.
- **Smart Stack Selection**: Presets for `DEMO`, `LOCAL`, `CLOUD` to manage dependencies.

### 🏗️ **Modular Architecture**
- **Component Registry**: Swap core components without changing code.
- **Configuration-Driven**: Control the stable RAG pipeline via JSON/YAML configs.

### 🎨 **Modern Frontend Interface**
- **Vue.js Application**: A simple web interface for demonstrating the chat functionality.

## 🚀 **Quick Start**

Given the experimental nature of the project, the most reliable way to get started is with the manual setup, focusing on the core components.

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

## 📚 **Documentation**

- [**🏗️ Project Structure**](PROJECT_STRUCTURE.md) - Understand the (newly updated) project layout.
- [**🧪 Testing Guide**](docs/guides/TESTING_GUIDE.md) - See how to run the (currently failing) test suite.
- [**🧩 Components**](docs/components/) - Documentation for the stable, core components.

---
*The rest of the README has been left as-is for now, but be aware that many of the features it describes under "Advanced Prompting", "Production-Ready Infrastructure", and "Conversational Routing" are the incomplete features mentioned above.*
