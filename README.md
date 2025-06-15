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

# For optional vector store providers
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install chromadb
pip install psycopg2-binary  # For PostgreSQL
pip install pinecone-client  # For Pinecone
pip install qdrant-client  # For Qdrant
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
  },  "vectorstore": {
    "provider": "faiss",  // Options: faiss, chroma, postgres, pinecone, qdrant
    "persist_directory": "./vector_store",
    "index_type": "Flat",
    "metric": "cosine"
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

This command will:
- Load documents from the specified paths
- Chunk them according to your chunking configuration
- Generate embeddings using the configured embedding provider
- Store the embeddings in your chosen vector store

### 3. Chat with your data

```bash
python -m rag_engine chat --config configs/your_config.json
```

You can also serve the engine as an API:

```bash
python -m rag_engine serve --config configs/your_config.json --api
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

## � Retrieval Strategies

RAG Engine offers multiple retrieval strategies to improve the quality and relevance of results:

### Simple Similarity Search

Basic vector similarity search, great for straightforward use cases:

```json
{
  "retrieval": {
    "retrieval_strategy": "simple",
    "top_k": 5
  }
}
```

### Threshold Retriever

Only retrieve documents above a similarity threshold:

```json
{
  "retrieval": {
    "retrieval_strategy": "threshold",
    "top_k": 10,
    "similarity_threshold": 0.75
  }
}
```

### Maximum Marginal Relevance (MMR)

Balance relevance with diversity in results:

```json
{
  "retrieval": {
    "retrieval_strategy": "mmr",
    "top_k": 5,
    "fetch_k": 20,
    "lambda_param": 0.7  // Higher values favor relevance over diversity
  }
}
```

### Hybrid Search

Combine dense vector search with keyword search for better results:

```json
{
  "retrieval": {
    "retrieval_strategy": "hybrid",
    "top_k": 5,
    "alpha": 0.7  // Weight for vector search vs keyword search
  }
}
```

### Reranker

Retrieve more candidates and rerank them:

```json
{
  "retrieval": {
    "retrieval_strategy": "rerank",
    "top_k": 5,
    "fetch_k": 20
  }
}
```

### Self-Query

Extract filters from the query to improve precision:

```json
{
  "retrieval": {
    "retrieval_strategy": "self_query",
    "top_k": 5,
    "metadata_schema": {
      "category": "string",
      "date": "string",
      "author": "string"
    }
  }
}
```

### Deduplication

Remove redundant or near-duplicate results:

```json
{
  "retrieval": {
    "retrieval_strategy": "dedup",
    "top_k": 5,
    "fetch_k": 15,
    "similarity_threshold": 0.85
  }
}
```

### Ensemble Retrieval

Combine multiple retrieval strategies:

```json
{
  "retrieval": {
    "retrieval_strategy": "ensemble",
    "top_k": 5,
    "strategy_weights": [0.7, 0.3]  // Weights for different strategies
  }
}
```

## �📊 Vector Store Options

RAG Engine supports multiple vector store providers for flexibility in different deployment scenarios:

### FAISS (Local In-Memory or Disk)

```json
{
  "vectorstore": {
    "provider": "faiss",
    "index_type": "Flat",  // Flat, IVF, HNSW
    "metric": "cosine",  // cosine, l2, euclidean
    "persist_directory": "./vector_stores/faiss"
  }
}
```

### ChromaDB (Local or Server)

```json
{
  "vectorstore": {
    "provider": "chroma",
    "collection_name": "rag_docs",
    "persist_directory": "./vector_stores/chroma",
    "metric": "cosine"
  }
}
```

For ChromaDB server:

```json
{
  "vectorstore": {
    "provider": "chroma",
    "host": "localhost",
    "port": 8000,
    "collection_name": "rag_docs"
  }
}
```

### PostgreSQL with pgVector

```json
{
  "vectorstore": {
    "provider": "postgres",
    "host": "localhost",
    "port": 5432,
    "database": "vector_db",
    "user": "postgres",
    "password": "${PGPASSWORD}",
    "table_name": "rag_documents",
    "metric": "cosine"  // cosine, l2, inner
  }
}
```

### Pinecone (Cloud)

```json
{
  "vectorstore": {
    "provider": "pinecone",
    "api_key": "${PINECONE_API_KEY}",
    "environment": "us-west1-gcp",
    "index_name": "rag-docs",
    "namespace": "default",
    "metric": "cosine"
  }
}
```

### Qdrant (Local or Cloud)

```json
{
  "vectorstore": {
    "provider": "qdrant",
    "collection_name": "rag_collection",
    "metric": "cosine",
    "url": "https://your-qdrant-cluster.cloud"  // For cloud deployment
  }
}
```

Local Qdrant:

```json
{
  "vectorstore": {
    "provider": "qdrant",
    "host": "localhost",
    "port": 6333,
    "collection_name": "rag_collection"
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
- **Vector Stores**: FAISS, ChromaDB, PostgreSQL/pgVector, Pinecone, Qdrant
- **Retrievers**: Simple similarity, MMR, hybrid search, reranker, ensemble
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

- [x] Complete Vector Store implementation (FAISS, ChromaDB, pgVector, Pinecone, Qdrant)
- [x] Retriever strategies (simple, MMR, hybrid, reranker, etc.)
- [ ] FastAPI server
- [ ] Streamlit/Gradio interface
- [ ] Evaluation metrics
- [ ] Pipeline versioning

## 📝 License

MIT License
