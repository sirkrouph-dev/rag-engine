# RAG Engine

A modular, highly customizable framework for building Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code.

## 🚀 Overview

RAG Engine is a plug-n-play framework that lets you customize every step of the RAG pipeline — from document loading to prompt engineering and LLM integration. Built with modularity in mind, it supports multiple interfaces (CLI, API, UI) and is designed for extensibility with a plugin system.

## ✨ Features

- **Full Customizability**: Configure every aspect of your RAG pipeline via YAML/JSON config files
- **Modular Architecture**: Swap components easily (loaders, chunkers, embedders, vector stores, LLMs)
- **Multiple Document Types**: Support for TXT, PDF, DOCX, and HTML documents
- **Advanced Chunking**: Fixed-size, sentence-based, semantic, and recursive chunking strategies
- **Flexible Embeddings**: OpenAI, Google Gemini/Vertex, SentenceTransformers
- **Flexible LLM Support**:
  - Cloud providers: OpenAI GPT models, Google Gemini
  - Local models: Phi-3, Gemma, and any model via Ollama
- **Multi-Interface**: Use via CLI, REST API, or web UI
- **Plugin System**: Extend with custom components

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-engine.git
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
    "provider": "openai",
    "model": "text-embedding-3-small",
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

## 📚 Document Processing

RAG Engine supports multiple document types:

### Document Loaders

```json
{
  "documents": [
    {"type": "pdf", "path": "./docs/sample.pdf"},
    {"type": "txt", "path": "./docs/notes.txt"},
    {"type": "docx", "path": "./docs/report.docx"},
    {"type": "html", "path": "./docs/webpage.html"}
  ]
}
```

### Chunking Strategies

Choose from multiple chunking strategies:

```json
{
  "chunking": {
    "method": "fixed",  // fixed, sentence, semantic, recursive
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

- **Fixed Size**: Chunk by character count
- **Sentence**: Chunk by natural sentence boundaries
- **Semantic**: Chunk by semantic elements (paragraphs, headers)
- **Recursive**: Smart recursive splitting with multiple separators

## 🔤 Embedding Options

RAG Engine supports multiple embedding providers:

### OpenAI Embeddings

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",  // or text-embedding-3-large
    "dimensions": 1536,  // Optional, reduce dimensions for efficiency
    "api_key": "${OPENAI_API_KEY}"
  }
}
```

### Google Gemini Embeddings

```json
{
  "embedding": {
    "provider": "gemini",
    "api_key": "${GOOGLE_API_KEY}"  // or from environment
  }
}
```

### Google Vertex AI Embeddings (Enterprise)

```json
{
  "embedding": {
    "provider": "gemini",
    "use_vertex": true,
    "project": "your-gcp-project-id",
    "location": "us-central1",
    "model": "textembedding-gecko@001"
  }
}
```

### Local Embeddings with SentenceTransformers

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-mpnet-base-v2",
    "normalize": true
  }
}
```

## 🤖 LLM Configuration

### OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",  // gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.
    "temperature": 0.3,
    "max_tokens": 1000,
    "api_key": "${OPENAI_API_KEY}",
    "system_prompt": "You are a helpful assistant."
  }
}
```

### Google Gemini

```json
{
  "llm": {
    "provider": "gemini",
    "model": "gemini-1.5-pro",  // or gemini-1.5-flash, gemini-1.0-pro, etc.
    "temperature": 0.3,
    "max_tokens": 1000,
    "api_key": "${GOOGLE_API_KEY}"
  }
}
```

### Local Models - Transformers (Direct)

```json
{
  "llm": {
    "provider": "local",
    "model_provider": "transformers", 
    "model": "microsoft/phi-3-mini",  // or google/gemma-7b
    "temperature": 0.7,
    "load_in_8bit": true,  // Quantization for efficiency
    "max_tokens": 1000
  }
}
```

### Local Models - Ollama

```json
{
  "llm": {
    "provider": "local",
    "model_provider": "ollama",
    "model": "phi3",  // or gemma:7b, llama3, mistral, etc.
    "temperature": 0.7,
    "ollama_host": "http://localhost:11434"
  }
}
```

## 🧩 Architecture

RAG Engine is built with a modular architecture:

- **Loaders**: PDF, DOCX, TXT, HTML
- **Chunkers**: Fixed-size, sentence-based, semantic, recursive
- **Embedders**: OpenAI, Gemini/Vertex, SentenceTransformers
- **Vector Stores**: Chroma, FAISS *(coming soon)*
- **LLMs**: OpenAI, Gemini, local models (Phi-3, Gemma, via Transformers or Ollama)
- **Interfaces**: CLI, API *(coming soon)*, UI *(coming soon)*

## 🧪 Extending RAG Engine

RAG Engine is designed to be extended with plugins. Create your custom components in the `rag_engine/plugins/` directory, implementing the appropriate base class:

```python
from rag_engine.core.base import BaseLoader

class MyCustomLoader(BaseLoader):
    def load(self, config):
        # Your custom logic here
        return documents
```

Then register in the appropriate registry:

```python
# In your plugin file
from rag_engine.core.loader import LOADER_REGISTRY

LOADER_REGISTRY["my_format"] = MyCustomLoader()
```

## 🔄 Development Roadmap

- [ ] Complete Vector Store implementation (Chroma, FAISS)
- [ ] Retriever strategies (top-k, MMR, filters)
- [ ] FastAPI server
- [ ] Streamlit/Gradio interface
- [ ] Evaluation metrics
- [ ] Pipeline versioning

## 📝 License

MIT License
