# RAG Engine

A modular, highly customizable framework for building Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code.

## 🚀 Overview

RAG Engine is a plug-n-play framework that lets you customize every step of the RAG pipeline — from document loading to prompt engineering and LLM integration. Built with modularity in mind, it supports multiple interfaces (CLI, API, UI) and is designed for extensibility with a plugin system.

## ✨ Features

- **Full Customizability**: Configure every aspect of your RAG pipeline via YAML/JSON config files
- **Modular Architecture**: Swap components easily (loaders, chunkers, embedders, vector stores, LLMs)
- **Multiple Document Types**: Support for TXT, PDF, DOCX, and HTML documents
- **Advanced Chunking**: Fixed-size, sentence-based, semantic, and recursive chunking strategies
- **Flexible LLM Support**:
  - Cloud providers: OpenAI GPT models, Google Gemini
  - Local models: Phi-3, Gemma, and any model via Ollama
- **Multi-Interface**: Use via CLI, REST API, or web UI
- **Plugin System**: Extend with custom components

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/sirkrouph-dev/rag-engine.git
cd rag-engine

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Quick Start

### 1. Create a config file (JSON or YAML)

```json
{
  "documents": [
    {"type": "pdf", "path": "./docs/sample.pdf"}
  ],
  "chunking": {
    "method": "recursive",
    "chunk_size": 512, 
    "chunk_overlap": 50
  },
  "embedding": {
    "model": "openai",
    "api_key": "${OPENAI_API_KEY}"
  },
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "./vector_store"
  },
  "retrieval": {
    "top_k": 4
  },
  "prompting": {
    "system_prompt": "You are a technical assistant. Answer clearly and concisely based on the provided context."
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.3
  }
}
```

### 2. Build your vector database

```bash
python -m rag_engine build --config configs/your_config.json
```

### 3. Chat with your data

```bash
python -m rag_engine chat --config configs/your_config.json
```

## 💡 Using Local LLMs

RAG Engine supports running local models like Phi-3 and Gemma:

### Option 1: Using Transformers (Direct)

```json
{
  "llm": {
    "provider": "local",
    "model_provider": "transformers", 
    "model": "microsoft/phi-3-mini",
    "temperature": 0.7,
    "load_in_8bit": true
  }
}
```

Requirements:
- Hugging Face account with model access
- Login via `huggingface-cli login` 
- Sufficient RAM (8GB+ recommended)
- GPU recommended but optional

### Option 2: Using Ollama (Easier)

1. Install Ollama from https://ollama.ai/
2. Pull your model: `ollama pull phi3`
3. Configure RAG Engine:

```json
{
  "llm": {
    "provider": "local",
    "model_provider": "ollama",
    "model": "phi3",
    "temperature": 0.7
  }
}
```

## 🧩 Architecture

RAG Engine is built with a modular architecture:

- **Loaders**: PDF, DOCX, TXT, HTML
- **Chunkers**: Fixed-size, sentence-based, semantic, recursive
- **Embedders**: OpenAI, HuggingFace, SentenceTransformers
- **Vector Stores**: Chroma, FAISS
- **LLMs**: OpenAI, Gemini, local models (Phi-3, Gemma)
- **Interfaces**: CLI, API, UI

## 🔄 Advanced Pipeline Configuration

For greater customization, you can configure each pipeline stage:

```json
{
  "chunking": {
    "method": "semantic",
    "html_elements": ["h1", "h2", "p"]
  },
  "embedding": {
    "model": "sentence-transformers/all-mpnet-base-v2", 
    "normalize": true
  },
  "retrieval": {
    "method": "mmr",
    "top_k": 5,
    "diversity": 0.3
  }
}
```

## 🧪 Extending RAG Engine

RAG Engine is designed to be extended with plugins. Create your custom components in the `rag_engine/plugins/` directory, implementing the appropriate base class.

## 📝 License

MIT License
