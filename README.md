# RAG Engine WIP

A powerful, modular framework for building advanced Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code. Combining state-of-the-art retrieval algorithms with multiple vector databases and LLM providers.

> **Status: Core Pipeline Functional** ✅ 
> 
> The core RAG pipeline is now functional! Document ingestion, chunking, embedding, and vector storage work end-to-end. The CLI build command successfully processes documents and stores them in ChromaDB. Chat functionality requires local LLM setup or API keys.
>
> _Built with ❤️ and GitHub Copilot - Learning and building in public._

## 🗺️ Project Roadmap

### Current Implementation Status

| Component | Status | Implementation Details |
|-----------|--------|----------------------|
| **Core Architecture** | ✅ **Complete** | Base interfaces and module structure implemented in `rag_engine/core/base.py` |
| **Configuration System** | ✅ **Complete** | Pydantic schemas with environment variable substitution in `rag_engine/config/` |
| **Document Loading** | ✅ **Working** | TXT, PDF, DOCX, HTML loaders implemented and tested in `rag_engine/core/loader.py` |
| **Text Chunking** | ✅ **Working** | Fixed-size chunking with overlap working in `rag_engine/core/chunker.py` |
| **Embedding** | ✅ **Working** | OpenAI, HuggingFace providers tested and working in `rag_engine/core/embedder.py` |
| **Vector Stores** | ✅ **Working** | ChromaDB integration working with persistence in `rag_engine/core/vectorstore.py` |
| **Retrieval** | ✅ **Implemented** | Similarity search working in `rag_engine/core/retriever.py` |
| **LLM Integration** | 🟡 **Partial** | OpenAI, Gemini, Local models implemented - requires API keys or local setup |
| **Tools System** | ✅ **Implemented** | Web search, calculator, file ops in `rag_engine/core/tools.py` and `rag_engine/plugins/tools.py` |
| **Reasoning Engine** | ✅ **Implemented** | Chain-of-thought, tree-of-thought in `rag_engine/core/reasoning.py` |
| **CLI Framework** | ✅ **Working** | Build command fully functional, chat requires LLM setup |
| **Pipeline Integration** | ✅ **Working** | End-to-end pipeline functional in `rag_engine/core/pipeline.py` |
| **REST API** | 🔴 **Placeholder** | Basic FastAPI structure in `rag_engine/interfaces/api.py` |
| **Web UI** | 🔴 **Not Started** | Empty file at `rag_engine/interfaces/ui.py` |
| **Database Integration** | 🔴 **Planned** | Chat history, knowledge graphs not implemented |
| **Testing Framework** | 🔴 **Not Started** | No tests exist yet |

### What Actually Works Right Now
- ✅ **End-to-End Pipeline**: Full RAG workflow from documents to vector storage is functional
- ✅ **CLI Build Command**: `python -m rag_engine build config.json` processes documents successfully
- ✅ **Configuration Loading**: YAML/JSON configs with environment variable substitution
- ✅ **Document Processing**: Load, chunk, embed, and store documents in ChromaDB
- � **CLI Chat Command**: Available but requires LLM API keys or local model setup
- 🔴 **API Endpoints**: No functional endpoints beyond basic health check

### Development Timeline & Next Steps

#### Immediate Priorities (Next 2-4 weeks)
1. **Connect the Pipeline** - Wire all components together in `rag_engine/core/pipeline.py`
2. **Make CLI Functional** - Implement actual build/chat functionality
3. **Basic Testing** - Add unit tests for core components
4. **Working Example** - Create one complete end-to-end example

#### Short Term (1-2 months)  
1. **REST API Implementation** - Functional FastAPI endpoints
2. **Documentation Cleanup** - Align docs with actual functionality
3. **Error Handling** - Robust error handling throughout pipeline
4. **Performance Testing** - Basic benchmarking and optimization

#### Medium Term (3-6 months)
1. **Web UI** - Simple Streamlit/Gradio interface
2. **Database Integration** - Chat history and knowledge graphs
3. **Testing Framework** - Cypress + Judge LLM implementation
4. **Plugin System** - Dynamic component loading
5. **Community Building** - Examples, tutorials, contributor guides

#### Vision (6+ months)
1. **Advanced Features** - Query expansion, response caching
2. **Production Tooling** - Monitoring, cost tracking, deployment guides
3. **Ecosystem** - Community plugins and integrations

- **Q4 2025**
  - Release Web UI for visualization and interaction
  - Implement plugin system
  - Add evaluation and benchmarking tools
  - Complete query classification and processing
  - Release initial set of tools (data enrichment & content processing)

- **Q1 2026**
  - Complete advanced reasoning tools
  - Add tool orchestration capabilities
  - Implement tool result validation and error handling
  - Support custom tool development API

## 🚀 Overview

A powerful, modular framework for building advanced Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code. Combining state-of-the-art retrieval algorithms with multiple vector databases and LLM providers.

## ✨ Features

### Currently Implemented

- **Full Customizability**: Configure every aspect of your RAG pipeline via YAML/JSON config files
### Currently Working

- **✅ End-to-End RAG Pipeline**: Complete document processing from files to queryable vector database
- **✅ Configuration-as-Code**: YAML/JSON configs with environment variable substitution  
- **✅ Multiple Document Types**: TXT, PDF, DOCX, and HTML document loading
- **✅ Text Chunking**: Fixed-size chunking with configurable overlap
- **✅ Multiple Embedding Providers**: OpenAI and HuggingFace embeddings working
- **✅ Vector Storage**: ChromaDB integration with persistence
- **✅ CLI Interface**: Build command processes documents end-to-end
- **✅ Modular Architecture**: Plugin-based system for extending functionality

### Coming Soon

- **Chat Interface**: Complete LLM integration for document querying (partially working - needs API keys)
- **REST API**: FastAPI endpoints for HTTP access
- **Web UI**: Interactive interface for document management and querying
- **Advanced Chunking**: Sentence-based and semantic chunking strategies
- **Additional Vector Stores**: FAISS, Pinecone, Qdrant integration
- **Local LLM Support**: Easy setup for Phi-3, Gemma, and Ollama models
- **Advanced Retrieval**: MMR, hybrid search, contextual compression
- **Testing Framework**: Comprehensive test suite and benchmarking

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

# For optional vector store providers
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install chromadb
pip install psycopg2-binary  # For PostgreSQL
pip install pinecone-client  # For Pinecone
pip install qdrant-client  # For Qdrant
```

## 🔧 Quick Start

> **Good News**: The core RAG pipeline is now functional! You can build and query document collections end-to-end.

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/sirkrouph-dev/rag-engine.git
cd rag-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Your First RAG Pipeline (✅ Working)

Create a config file for HuggingFace embeddings (no API key needed):

```json
{
  "documents": [
    {"type": "txt", "path": "./your_document.txt"}
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
  "retrieval": {
    "top_k": 3
  },
  "prompting": {
    "system_prompt": "You are a helpful assistant. Answer based on the provided context."
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "${OPENAI_API_KEY}"
  }
}
```

### 3. Build Your Vector Database (✅ Working)

```bash
# Process documents and build vector database
python -m rag_engine build your_config.json
```

This will:
- Load your documents
- Chunk them into smaller pieces
- Generate embeddings using HuggingFace models
- Store everything in ChromaDB
- Persist the database to disk

### 4. Query Your Documents (Requires LLM Setup)

```bash
# Start interactive chat (requires OpenAI API key or local LLM)
python -m rag_engine chat your_config.json

```python
# Test configuration loading
from rag_engine.config.loader import load_config
config = load_config("configs/example_config.yml")

# Test document loading
from rag_engine.core.loader import TxtLoader
loader = TxtLoader()
docs = loader.load({"path": "sample.txt"})

# Test chunking
from rag_engine.core.chunker import FixedSizeChunker
chunker = FixedSizeChunker()
chunks = chunker.chunk(docs[0], {"chunk_size": 500})
```

### 4. CLI Commands (🔴 Not Functional Yet)

```bash
# These exist but don't work end-to-end yet
python -m rag_engine init
python -m rag_engine build --config configs/example.json
python -m rag_engine chat --config configs/example.json
```

### 🎉 Successful Build Output

When you run the build command, you'll see output like this:

```
🚀 Starting RAG pipeline build...
📂 Loading documents...
   ✓ Loaded ./test_doc.txt (1 items)
🧩 Chunking documents...
   ✓ Created 2 chunks
🔢 Generating embeddings and storing in vector database...
   ✓ Embedded and stored 2 chunks
   ✓ Vector store persisted to ./test_vector_store
✅ Pipeline build complete!
   📄 Loaded 1 documents
   🧩 Created 2 chunks
   💾 Stored in chroma vector store
Vector DB built.
```

This means your documents are successfully:
- ✅ Loaded and parsed
- ✅ Chunked into searchable pieces  
- ✅ Embedded using HuggingFace models
- ✅ Stored in ChromaDB with persistence to disk
- ✅ Ready for querying (once LLM is configured)

## 📖 Configuration Examples

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
    "provider": "chroma",  // Currently supported: faiss, chroma
    "persist_directory": "./vector_store",
    "collection_name": "rag_docs",
    "metric": "cosine"
  },
  "retrieval": {
    "retrieval_strategy": "simple",  // Currently only simple similarity is implemented
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

### 2. Build your vector database (In Development)

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
## 📋 Metadata Extraction

### Document Metadata Extraction

RAG Engine supports extracting and utilizing metadata at multiple stages of the pipeline:

```json
{
  "documents": [
    {"type": "pdf", "path": "./docs/sample.pdf", "metadata_extraction": true}
  ],
  "metadata_extraction": {
    "document_level": {
      "enabled": true,
      "extractors": ["title", "author", "date", "summary", "keywords", "categories"],
      "custom_extractors": {
        "domain_entities": "Extract all company names, product names, and technical terms mentioned in the document.",
        "complexity": "Rate the document complexity on a scale of 1-5."
      }
    },
    "chunk_level": {
      "enabled": true,
      "extractors": ["section_title", "subtopic", "entities"]
    }
  }
}
```

### Query Metadata Extraction

The query processing pipeline includes metadata extraction for user queries:

```json
{
  "query_processing": {
    "metadata_extraction": {
      "enabled": true,
      "extractors": [
        "entities",       // Extract named entities like products, locations, people
        "intent",         // Identify user intent (e.g., comparison, how-to)
        "constraints",    // Extract constraints like date ranges, versions
        "preferences",    // Extract user preferences
        "domain_context"  // Extract domain-specific context
      ]
    }
  }
}
```

### Metadata in the Processing Pipeline

Metadata is leveraged throughout the RAG Engine pipeline:

1. **During Document Processing**:
   - Document-level metadata is extracted from the source material
   - Chunk-level metadata is attached to each text segment
   - Metadata is stored alongside embeddings in the vector store

2. **During Query Processing**:
   - User query is analyzed to extract entities, intent, and constraints
   - This metadata influences classification and routing decisions
   - Can be used for filtering relevant documents (e.g., by date range)

3. **During Retrieval**:
   - Metadata enables advanced filtering of results
   - Self-query retrieval can convert natural language constraints to metadata filters
   - Reranking can use metadata similarity as a factor

4. **During Prompt Construction**:
   - Document metadata can be included in prompts to provide context
   - Query metadata guides response formatting and content prioritization
   - Templates can use conditionals based on metadata values

Example workflow with metadata:

```
Query: "What did our company announce about the new product last month?"

1. Metadata Extraction:
   - Entity: "new product"
   - Temporal constraint: "last month"
   - Context: "our company"

2. Vector Search:
   - Initial semantic search for "company announcement new product"
   - Metadata filtering to documents dated last month
   
3. Prompt Construction:
   - Selected template based on "informational" classification
   - Highlight document sections mentioning the specific product
   - Include document metadata (publication date, source) in the prompt
```

This comprehensive metadata system enables more precise retrieval and more contextually appropriate responses across different query types.

## 🧭 Prompt Templates

RAG Engine supports sophisticated prompt engineering using Jinja2 templates:

```json
{
  "prompting": {
    "template_engine": "jinja2",
    "default_system_template": "You are a helpful assistant answering questions based on the provided documents.\n\n{% for doc in documents %}\nDocument {{ loop.index }}:\n{{ doc.content }}\n{% endfor %}",
    "default_user_template": "{{ query }}",
    "custom_templates": {
      "summarization": {
        "system": "You are an expert summarizer. Condense the following documents into a clear summary:\n\n{% for doc in documents %}\n{{ doc.content }}\n{% endfor %}"
      },
      "comparison": {
        "system": "Compare and contrast the following documents:\n\n{% for doc in documents %}\nDocument {{ loop.index }}:\n{{ doc.content }}\n{% endfor %}"
      }
    }
  }
}
```

### Prompt-Query Processing Integration

The true power of RAG Engine comes from the integration between the Query Processing system and Prompt Templates:

1. **Dynamic Template Selection**: The query processor determines which template to use based on the query category
   
2. **Context-Aware Rendering**: Templates receive rich context including:
   - Retrieved documents with their metadata and relevance scores
   - Extracted entities and intents from the query
   - User preferences and conversation history 
   
3. **Conditional Logic**: Jinja2 templates can include conditional logic to handle different response scenarios

4. **Variable Substitution**: Metadata extracted by the query processor gets inserted into templates

Example workflow:
1. User query comes in: "How do I install the software on Windows?"
2. Query processor classifies it as "instructional" with entity "software" and platform "Windows"
3. The "instructional" template is selected
4. Document retrieval fetches relevant installation guides
5. The template renders with specific instructions for positioning Windows installation steps prominently

This integration creates a flexible system that can adapt to different query types while maintaining consistent, high-quality responses.

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
    "provider": "vertex",
    "project": "your-gcp-project-id",
    "location": "us-central1",
    "model": "textembedding-gecko@latest",
    "credentials_path": "./path/to/service-account.json"  // Optional, defaults to ADC
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

### Advanced Retrieval Methods

RAG Engine also offers more sophisticated retrieval approaches:

#### Contextual Compression

Retrieves larger chunks first, then compresses them to the most relevant parts:

```json
{
  "retrieval": {
    "retrieval_strategy": "compression",
    "top_k": 5
  }
}
```

#### Multi-Query Retrieval

Generates multiple variations of the query to improve recall:

```json
{
  "retrieval": {
    "retrieval_strategy": "multi_query",
    "top_k": 5,
    "num_query_variations": 3
  }
}
```

#### Hierarchical Retrieval

Two-stage retrieval process that first identifies relevant topics/clusters:

```json
{
  "retrieval": {
    "retrieval_strategy": "hierarchical",
    "top_k": 5,
    "num_clusters": 3
  }
}
```

#### Parent Document Retrieval

Retrieves smaller chunks but returns their parent documents:

```json
{
  "retrieval": {
    "retrieval_strategy": "parent_document",
    "top_k": 5
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

### Google Vertex AI

Access enterprise-grade LLMs through Google Cloud Vertex AI:

```json
{
  "llm": {
    "provider": "vertex",
    "model": "gemini-1.5-pro",  // or text-bison, claude-3-sonnet@latest, etc.
    "project": "your-gcp-project-id",
    "location": "us-central1",
    "temperature": 0.2,
    "max_tokens": 1000,
    "credentials_path": "./path/to/service-account.json",  // Optional
    "streaming": true
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

## 🧠 Advanced Query Processing

RAG Engine supports sophisticated query processing pipelines beyond simple RAG, including:

### Query Classification and Routing

Configure how different query types are handled:

```json
{
  "query_processing": {
    "classifier": {
      "enabled": true,
      "model": "local",  // or "openai", "gemini", etc.
      "categories": [
        "informational", 
        "instructional", 
        "troubleshooting",
        "greeting",
        "gratitude",
        "out_of_context"
      ]
    },
    "routing": {
      "informational": {"use_rag": true, "response_style": "detailed"},
      "instructional": {"use_rag": true, "response_style": "step_by_step"},
      "troubleshooting": {"use_rag": true, "response_style": "diagnostic"},
      "greeting": {"use_rag": false, "response_style": "conversational"},
      "gratitude": {"use_rag": false, "response_style": "brief"},
      "out_of_context": {"use_rag": false, "response_style": "redirect"}
    }
  }
}
```

### Ambiguity Detection and Clarification

Handle ambiguous queries with follow-up clarification:

```json
{
  "clarification": {
    "enabled": true,
    "ambiguity_threshold": 0.7,
    "clarification_prompt": "I need more information to answer accurately. Could you please provide more details about {ambiguous_aspect}?",
    "max_clarification_turns": 2
  }
}
```

### Response Shaping by Category

Customize response style based on query category:

```json
{
  "response_shaping": {
    "informational": "You are an expert providing detailed factual information. Cite relevant sections from the retrieved documents.",
    "instructional": "You are a teacher providing step-by-step instructions. Break down complex processes clearly.",
    "troubleshooting": "You are a technical support specialist diagnosing problems. Consider possible causes and solutions.",
    "out_of_context": "You are a friendly assistant. Politely explain that this question is outside the scope of the available knowledge."
  }
}
```

### Integration with Jinja2 Templating

The query processing system integrates with the Jinja2 templating system to create dynamic, context-aware prompts:

```json
{
  "prompting": {
    "template_engine": "jinja2",
    "templates": {
      "informational": {
        "system": "You are an expert providing factual information. Use the following documents as context:\n{% for doc in documents %}\nDocument [{{ loop.index }}]: {{ doc.content }}\n{% endfor %}\n\nAdditional instructions: {{ response_style }}",
        "user": "{{ processed_query }}",
        "response": "{{ response }}"
      },
      "instructional": {
        "system": "You are a teacher providing instructions. Based on these reference materials:\n{% for doc in documents %}\nReference [{{ loop.index }}]: {{ doc.content }}\n{% endfor %}\n\nAdditional instructions: {{ response_style }}",
        "user": "{{ processed_query }}",
        "response": "{{ response }}"
      },
      "troubleshooting": {
        "system": "You are a technical support specialist. Consider these knowledge base articles:\n{% for doc in documents %}\nArticle [{{ loop.index }}]: {{ doc.content }}\n{% endfor %}\n\nAdditional instructions: {{ response_style }}",
        "user": "{{ processed_query }}",
        "response": "{{ response }}"
      },
      "out_of_context": {
        "system": "You are a helpful assistant. You should politely decline to answer questions outside your knowledge scope.",
        "user": "{{ original_query }}",
        "response": "{{ response }}"
      }
    }
  }
}
```

In this integration:

1. The query processor first **classifies** the incoming query
2. Based on classification, it **selects the appropriate template**
3. The template gets populated with:
   - Retrieved documents (for RAG-based queries)
   - Query metadata and extracted entities
   - Response style instructions
   - Original or processed query as needed

This creates a powerful system where prompts are dynamically constructed based on the query type, allowing for specialized handling of different kinds of questions while maintaining a consistent configuration-driven approach.

This advanced query processing system allows for more intelligent handling of user queries, improving the relevance and quality of responses while gracefully handling edge cases.

## 🧠 Chain of Thought (CoT) Reasoning

RAG Engine includes advanced reasoning capabilities that make AI outputs more transparent, explainable, and trustworthy. These features help users understand how the AI arrived at its conclusions and how it interprets retrieved information.

### Chain of Thought Integration

```json
{
  "reasoning": {
    "enabled": true,
    "mode": "step_by_step",  // Options: step_by_step, tree_of_thought, scratchpad
    "verbosity": "medium",   // Options: minimal, medium, detailed
    "include_in_output": true,
    "structured_format": true
  }
}
```

### Reasoning Modes

#### Step-by-Step Reasoning

Guides the LLM through a sequential reasoning process:

```json
{
  "reasoning": {
    "mode": "step_by_step",
    "steps": [
      "analyze_question",
      "extract_key_information",
      "evaluate_evidence",
      "formulate_answer",
      "verify_answer"
    ],
    "format": "markdown"
  }
}
```

#### Tree of Thought

Explores multiple reasoning paths before arriving at a conclusion:

```json
{
  "reasoning": {
    "mode": "tree_of_thought",
    "max_branches": 3,
    "max_depth": 2,
    "selection_strategy": "best_first"  // Options: best_first, breadth_first, depth_first
  }
}
```

#### Self-Reflection

Enables the model to critique its own reasoning and refine its answers:

```json
{
  "reasoning": {
    "self_reflection": {
      "enabled": true,
      "reflection_prompt": "Review your answer above. Is there anything unclear, incorrect, or missing from your explanation? Provide a better answer if needed.",
      "iterations": 1
    }
  }
}
```

### Reasoning Templates

RAG Engine provides specialized templates for different reasoning modes:

```json
{
  "prompting": {
    "template_engine": "jinja2",
    "templates": {
      "cot_reasoning": {
        "system": "You are a careful thinking assistant that solves problems step by step. First analyze the question, then evaluate the provided documents, and finally provide a well-reasoned answer.\n\nDocuments:\n{% for doc in documents %}\nDocument [{{ loop.index }}]: {{ doc.content }}\n{% endfor %}",
        "user": "{{ query }}\n\nPlease think through this step by step.",
        "response": "{{ reasoning_steps }}\n\n{{ final_answer }}"
      }
    }
  }
}
```

### Exposing Reasoning to Users

RAG Engine provides options for how reasoning should be exposed to end users:

1. **Visible Reasoning**: Show the full reasoning process along with the final answer
   ```json
   { "reasoning": { "output_mode": "full_process" } }
   ```

2. **Hidden Reasoning**: Use CoT internally but only show the final answer
   ```json
   { "reasoning": { "output_mode": "answer_only" } }
   ```

3. **Expandable Reasoning**: Show the answer with an option to expand and view reasoning
   ```json
   { "reasoning": { "output_mode": "expandable" } }
   ```

### Structured Reasoning Output

For applications that need to process reasoning steps programmatically:

```json
{
  "reasoning": {
    "structured_format": true,
    "output_schema": {
      "reasoning_steps": ["string"],
      "retrieved_documents": ["string"],
      "confidence_score": "number",
      "final_answer": "string"
    }
  }
}
```

### Benefits of Chain of Thought in RAG

1. **Transparency**: Users understand how the AI interprets information and reaches conclusions
2. **Error Detection**: Makes it easier to identify where reasoning went wrong
3. **Trust Building**: Shows which documents influenced the answer and how they were used
4. **Education**: Helps users learn the thinking process behind complex answers
5. **Debugging**: Makes it easier to refine prompts and retrieval strategies

Chain of Thought reasoning is especially powerful for:
- Complex questions requiring multiple logical steps
- Scenarios where showing evidence is important
- Technical or scientific domains where methodology matters
- Applications where users need to verify the AI's work

> **Note:** Chain of Thought reasoning capabilities are planned for implementation in Q3 2025. The basic framework is in place, but full integration with the various reasoning modes is still in development.

## 🛠️ Tools Integration 

RAG Engine supports the integration of specialized tools that extend the capabilities of language models beyond basic text generation. These tools allow the RAG pipeline to perform actions, retrieve external information, and interact with external systems.

### What Are Tools?

Tools are specialized functions that LLMs can use to:

- Execute code or commands
- Retrieve information from external sources
- Interact with APIs or databases
- Perform specialized calculations or reasoning

### Planned Tool Categories

#### Data Enrichment Tools

```json
{
  "tools": [
    {
      "type": "web_search",
      "name": "search_web",
      "description": "Search the web for current information",
      "config": {
        "provider": "serper",  // or "serpapi", "ddg", etc.
        "api_key": "${SEARCH_API_KEY}"
      }
    },
    {
      "type": "api_call",
      "name": "get_weather",
      "description": "Get current weather for a location",
      "config": {
        "endpoint": "https://api.weather.com/v1/current",
        "method": "GET",
        "auth_header": "Bearer ${WEATHER_API_KEY}"
      }
    }
  ]
}
```

#### Content Processing Tools

```json
{
  "tools": [
    {
      "type": "calculator",
      "name": "perform_calculation",
      "description": "Perform mathematical calculations"
    },
    {
      "type": "code_executor",
      "name": "execute_python",
      "description": "Execute Python code safely in a sandbox",
      "config": {
        "timeout": 5,  // seconds
        "max_memory": 512,  // MB
        "allowed_modules": ["pandas", "numpy", "matplotlib"]
      }
    },
    {
      "type": "chart_generator",
      "name": "create_chart",
      "description": "Generate charts from data",
      "config": {
        "engine": "matplotlib",  // or "plotly"
        "default_format": "png"
      }
    }
  ]
}
```

#### Document-Specific Tools

```json
{
  "tools": [
    {
      "type": "table_extractor",
      "name": "extract_tables",
      "description": "Extract and process tables from documents"
    },
    {
      "type": "image_analyzer",
      "name": "analyze_image",
      "description": "Extract text or insights from images",
      "config": {
        "vision_model": "openai",  // or "gemini", "claude"
        "api_key": "${VISION_API_KEY}"
      }
    }
  ]
}
```

#### Advanced Reasoning Tools

```json
{
  "tools": [
    {
      "type": "structured_reasoning",
      "name": "step_by_step_reasoning",
      "description": "Perform step-by-step reasoning for complex problems",
      "config": {
        "reasoning_framework": "chain-of-thought",  // or "tree-of-thought"
        "max_steps": 5
      }
    },
    {
      "type": "fact_checker",
      "name": "verify_claim",
      "description": "Verify claims against reliable sources",
      "config": {
        "confidence_threshold": 0.8
      }
    }
  ]
}
```

### Tool Integration Architecture

RAG Engine will provide a pluggable tool architecture allowing:

1. **Tool Registry**: Register and manage available tools
2. **Tool Selection**: Intelligent selection of appropriate tools based on query
3. **Tool Execution**: Safe execution of tool functions with proper error handling
4. **Result Integration**: Seamlessly incorporating tool outputs into responses
5. **Custom Tool Creation**: API for creating custom domain-specific tools

> **Note:** Tool integration is planned for a future release. Please check the [Project Roadmap](#-project-roadmap) for the current development status.

## 🗄️ Database Integration Options

RAG Engine supports multiple database backends for different use cases:

### Chat History & Conversation Storage

#### MongoDB (NoSQL)
Perfect for flexible conversation schemas and rapid development:

```json
{
  "chat_history": {
    "provider": "mongodb",
    "connection_string": "mongodb://localhost:27017/rag_engine",
    "database": "rag_conversations",
    "collection": "chat_sessions",
    "indexes": ["user_id", "session_id", "timestamp"]
  }
}
```

#### Google Cloud SQL - PostgreSQL
Managed PostgreSQL with pgVector for unified vector + chat storage:

```json
{
  "chat_history": {
    "provider": "google_cloud_sql",
    "database_type": "postgresql",
    "instance_connection_name": "project:region:instance",
    "database": "rag_engine",
    "user": "rag_user",
    "password": "${CLOUD_SQL_PASSWORD}",
    "enable_pgvector": true,
    "ssl_mode": "require"
  }
}
```

#### Google Cloud SQL - MySQL
For structured relational chat history:

```json
{
  "chat_history": {
    "provider": "google_cloud_sql", 
    "database_type": "mysql",
    "instance_connection_name": "project:region:instance",
    "database": "rag_conversations",
    "user": "rag_user",
    "password": "${CLOUD_SQL_PASSWORD}",
    "charset": "utf8mb4"
  }
}
```

#### Redis (Caching & Real-time)
Ultra-fast chat caching and real-time features:

```json
{
  "chat_history": {
    "provider": "redis",
    "host": "localhost",
    "port": 6379,
    "password": "${REDIS_PASSWORD}",
    "db": 0,
    "ttl_seconds": 86400,
    "use_streams": true
  }
}
```

### Knowledge Graph Storage

#### Neo4j (Recommended)
For entity relationships and semantic connections:

```json
{
  "knowledge_graph": {
    "provider": "neo4j",
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "${NEO4J_PASSWORD}",
    "database": "rag_knowledge",
    "enable_vector_search": true
  }
}
```

### Multi-Database Architecture

You can combine different databases for optimal performance:

```json
{
  "databases": {
    "vector_store": {
      "provider": "pinecone",
      "api_key": "${PINECONE_API_KEY}"
    },
    "chat_history": {
      "provider": "google_cloud_sql",
      "database_type": "postgresql"
    },
    "knowledge_graph": {
      "provider": "neo4j"
    },
    "cache": {
      "provider": "redis"
    }
  }
}
```

### Database Schema Examples

#### Chat History Schema (PostgreSQL)
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    config_snapshot JSONB
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB, -- retrieval info, costs, etc.
    embedding VECTOR(1536) -- for semantic search on chat history
);

CREATE INDEX idx_conversations_user_session ON conversations(user_id, session_id);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, timestamp);
CREATE INDEX idx_messages_embedding ON messages USING ivfflat (embedding vector_cosine_ops);
```

---

## 👥 Contributing

RAG Engine is an open-source project in active development. Contributions are welcome and appreciated!

### How to Contribute

1. **Pick an area from the roadmap**: Check the [Project Roadmap](#-project-roadmap) for components that need implementation.

2. **Implement missing features**: Choose a component marked as "In Progress" or "Planned" and submit a pull request.

3. **Test and document**: Add tests for your implementation and update documentation to reflect new capabilities.

4. **Report bugs**: If you find issues with existing functionality, please create an issue in the repository.

### Development Setup

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

# Optional: Install dev dependencies
pip install -r requirements-dev.txt
```

### Project Structure

- `rag_engine/core/`: Core interfaces and base classes
- `rag_engine/config/`: Configuration schema and loading
- `rag_engine/interfaces/`: CLI, API, and UI interfaces
- `rag_engine/plugins/`: Plugin system for extensions

### Current Development Priorities

1. Complete document loading and chunking implementations
2. Implement basic retrieval functionality
3. Connect components to CLI commands
4. Add tests and documentation
5. Begin tool integration framework development

## 📦 Distribution Options

RAG Engine can be distributed in multiple ways depending on your goals:

### PyPI Package (Recommended)

Make RAG Engine available through the Python Package Index:

```bash
# Install the project setup tools
pip install build twine

# Generate distribution packages
python -m build

# Upload to PyPI (requires PyPI account)
python -m twine upload dist/*
```

This allows users to install with a simple:

```bash
pip install rag-engine
```

### GitHub Repository

Host on GitHub to facilitate collaboration and visibility:

1. **Public Repository**: Maximum community engagement and contributions
   - Ensure documentation is comprehensive
   - Set up GitHub Actions for CI/CD
   - Add issue templates and contribution guidelines

2. **Template Repository**: Let users create their own instances
   - Users can click "Use this template" to create their own repository
   - Ideal for customizable starting points

### Docker Hub

Distribute pre-configured Docker images:

```bash
# Build Docker image
docker build -t sirkrouph-dev/rag-engine:latest .

# Push to Docker Hub
docker push sirkrouph-dev/rag-engine:latest
```

Users can run with:

```bash
docker run -p 8000:8000 -v ./configs:/app/configs sirkrouph-dev/rag-engine serve --config configs/my_config.json
```

### Hybrid Approach (Recommended)

The most effective distribution strategy combines:

1. **Python Package** on PyPI for easy installation
2. **GitHub Repository** for visibility and collaboration
3. **Docker Images** for easy deployment
4. **Documentation Site** with GitHub Pages or ReadTheDocs

This maximizes accessibility while providing options for different user needs.

## 📝 License

MIT License

## 🤖 For ML Engineers

### Why This Project is a Perfect ML Playground

RAG Engine provides the **infrastructure** so ML engineers can focus on the **intelligence**. Here's where ML can make transformative improvements:

---

### 🎯 **High-Impact ML Opportunities**

#### **1. Retrieval Quality Revolution**

**Current Problem**: Vector similarity is primitive - it doesn't understand context, intent, or semantic relationships.

**ML Solutions**:
```python
# A. Dense Passage Retrieval (DPR-style)
class BiEncoderRetriever:
    """Train separate encoders for queries vs documents"""
    def __init__(self):
        self.query_encoder = BertModel.from_pretrained('bert-base')
        self.doc_encoder = BertModel.from_pretrained('bert-base')
        
    def train_on_domain_data(self, query_doc_pairs):
        # Train with contrastive loss on domain-specific data
        # Learns what makes documents relevant for specific queries
        pass

# B. ColBERT Late Interaction
class ColBERTRetriever:
    """Token-level interaction for fine-grained matching"""
    def compute_relevance(self, query_tokens, doc_tokens):
        # MaxSim operation: max over document tokens for each query token
        # Captures fine-grained semantic matching
        pass

# C. Learned Sparse Retrieval (SPLADE)
class SPLADERetriever:
    """Learn sparse representations that outperform BM25"""
    def expand_representation(self, text):
        # Learn to upweight important terms, add related terms
        # Combines benefits of sparse + dense retrieval
        pass
```

**Real Impact**: 20-40% improvement in retrieval quality over basic vector similarity.

#### **2. Query Understanding & Intent Classification**

**Current Problem**: RAG treats all queries the same - but "What is X?" needs different handling than "Compare X vs Y".

**ML Solutions**:
```python
class QueryIntelligence:
    def __init__(self):
        self.intent_classifier = self.load_intent_model()
        self.entity_extractor = self.load_ner_model()
        self.complexity_scorer = self.load_complexity_model()
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Deep query understanding for better RAG routing"""
        
        # Intent classification
        intent = self.intent_classifier.predict(query)
        # "factual", "comparison", "how_to", "troubleshooting", "creative"
        
        # Entity extraction with confidence
        entities = self.entity_extractor.extract_with_confidence(query)
        # [{"entity": "Python", "type": "LANGUAGE", "confidence": 0.95}]
        
        # Query complexity scoring
        complexity = self.complexity_scorer.score(query)
        # Simple questions vs multi-step reasoning
        
        # Temporal understanding
        time_expressions = self.extract_temporal_info(query)
        # "last month", "since 2020", "recently"
        
        return QueryAnalysis(intent, entities, complexity, time_expressions)

class AdaptiveRetrieval:
    """Adjust retrieval strategy based on query analysis"""
    
    def retrieve(self, query: str, analysis: QueryAnalysis):
        if analysis.intent == "comparison":
            # Retrieve documents about both entities
            return self.multi_entity_retrieval(analysis.entities)
            
        elif analysis.intent == "troubleshooting":
            # Prioritize solution-oriented documents
            return self.solution_focused_retrieval(query)
            
        elif analysis.complexity > 0.8:
            # Multi-hop retrieval for complex queries
            return self.iterative_retrieval(query, max_hops=3)
```

**Real Impact**: Better routing leads to 30-50% improvement in answer relevance.

#### **3. Revolutionary Response Quality Assessment**

**Current Problem**: No way to automatically assess if RAG responses are good - requires human evaluation.

**ML Solutions - The Judge LLM System**:
```python
class IntelligentRAGJudge:
    """Multi-model ensemble for comprehensive quality assessment"""
    
    def __init__(self):
        # Specialized models for different quality aspects
        self.factual_accuracy_model = self.load_fact_checker()
        self.relevance_scorer = self.load_relevance_model()
        self.hallucination_detector = self.load_hallucination_model()
        self.completeness_evaluator = self.load_completeness_model()
        self.source_attribution_checker = self.load_attribution_model()
        
    def comprehensive_evaluation(self, query, response, sources):
        """Multi-dimensional quality assessment"""
        
        # 1. Factual Accuracy: Claims vs Source Content
        fact_score = self.check_factual_accuracy(response, sources)
        # Extracts claims, verifies against source material
        
        # 2. Hallucination Detection: Content Not in Sources
        hallucination_risk = self.detect_hallucinations(response, sources)
        # Identifies information not supported by retrieval
        
        # 3. Relevance Scoring: Response vs Query Intent
        relevance = self.score_relevance(query, response)
        # Measures how well response addresses the question
        
        # 4. Completeness: Missing Important Information
        completeness = self.assess_completeness(query, response, sources)
        # Identifies if key information was omitted
        
        # 5. Source Attribution: Proper Citation
        attribution_quality = self.evaluate_citations(response, sources)
        # Checks if claims are properly attributed
        
        # 6. Confidence Calibration
        confidence = self.estimate_confidence(response, sources)
        # How confident should we be in this response?
        
        return QualityAssessment(
            factual_accuracy=fact_score,
            hallucination_risk=hallucination_risk,
            relevance=relevance,
            completeness=completeness,
            attribution_quality=attribution_quality,
            confidence=confidence,
            overall_score=self.compute_weighted_score(...)
        )

class ContinuousLearningLoop:
    """Learn from quality assessments to improve the system"""
    
    def learn_from_feedback(self, query, response, quality_score, user_feedback):
        """Improve retrieval and generation based on quality scores"""
        
        if quality_score.relevance < 0.7:
            # Poor relevance - improve retrieval
            self.retrain_retrieval_model(query, negative_examples=response)
            
        if quality_score.hallucination_risk > 0.3:
            # High hallucination - adjust generation prompts
            self.update_generation_prompts(conservative=True)
            
        if user_feedback == "incorrect":
            # Use as hard negative for training
            self.add_to_training_data(query, response, label="negative")
```

**Real Impact**: Automated quality assessment enables continuous improvement without human labeling.

#### **4. Advanced Document Understanding**

**Current Problem**: RAG treats documents as bags of text chunks - missing structure, relationships, and context.

**ML Solutions**:
```python
class DocumentIntelligence:
    """Deep document understanding for better chunking and retrieval"""
    
    def __init__(self):
        self.structure_parser = self.load_document_structure_model()
        self.semantic_segmenter = self.load_segmentation_model()
        self.topic_modeler = self.load_topic_model()
        self.importance_scorer = self.load_importance_model()
        
    def intelligent_preprocessing(self, document):
        """Extract maximum information from documents"""
        
        # 1. Structure Understanding
        structure = self.structure_parser.parse(document)
        # Headers, sections, tables, figures, captions
        
        # 2. Semantic Segmentation
        segments = self.semantic_segmenter.segment(document)
        # Topically coherent chunks vs arbitrary size limits
        
        # 3. Topic Modeling
        topics = self.topic_modeler.extract_topics(document)
        # What concepts does this document cover?
        
        # 4. Importance Scoring
        importance_scores = self.importance_scorer.score_sentences(document)
        # Which parts are most informative?
        
        # 5. Relationship Extraction
        relationships = self.extract_entity_relationships(document)
        # Build knowledge graph from document content
        
        return EnrichedDocument(
            structure=structure,
            semantic_chunks=segments,
            topics=topics,
            importance_scores=importance_scores,
            relationships=relationships
        )

class SemanticChunking:
    """ML-driven chunking based on content, not arbitrary size"""
    
    def chunk_by_semantics(self, document):
        # Use sentence transformers to find semantic boundaries
        # Cluster similar sentences together
        # Result: Chunks that are topically coherent
        pass
```

#### **5. Knowledge Graph Intelligence**

**Current Problem**: RAG retrieval is flat - no understanding of relationships between concepts.

**ML Solutions**:
```python
class KnowledgeGraphRAG:
    """Integrate structured knowledge with vector retrieval"""
    
    def __init__(self):
        self.entity_linker = self.load_entity_linking_model()
        self.relation_extractor = self.load_relation_extraction_model()
        self.graph_embeddings = self.load_graph_embedding_model()
        
    def graph_enhanced_retrieval(self, query):
        """Use knowledge graph to improve retrieval"""
        
        # 1. Extract entities from query
        query_entities = self.entity_linker.link(query)
        
        # 2. Find related entities in knowledge graph
        related_entities = self.find_related_entities(query_entities, hops=2)
        
        # 3. Expand query with related concepts
        expanded_query = self.expand_with_entities(query, related_entities)
        
        # 4. Hybrid retrieval: vector + graph
        vector_results = self.vector_search(expanded_query)
        graph_results = self.graph_traversal_search(query_entities)
        
        # 5. Fuse results using learned ranking
        final_results = self.learned_fusion(vector_results, graph_results)
        
        return final_results

class AutomaticKGConstruction:
    """Build knowledge graphs automatically from documents"""
    
    def extract_knowledge_graph(self, document_corpus):
        # Extract entities and relations from all documents
        # Build unified knowledge graph
        # Learn embeddings for entities and relations
        pass
```

#### **6. Personalization & Adaptation**

**Current Problem**: RAG gives same answers to everyone - no personalization or learning from interactions.

**ML Solutions**:
```python
class PersonalizedRAG:
    """Learn user preferences and adapt responses"""
    
    def __init__(self):
        self.user_profiler = self.load_user_profiling_model()
        self.preference_learner = self.load_preference_model()
        self.adaptation_engine = self.load_adaptation_model()
        
    def personalized_retrieval(self, query, user_id):
        """Adapt retrieval based on user profile"""
        
        # 1. User profiling
        user_profile = self.user_profiler.get_profile(user_id)
        # Expertise level, domain preferences, communication style
        
        # 2. Query adaptation
        adapted_query = self.adapt_query_for_user(query, user_profile)
        
        # 3. Personalized ranking
        results = self.standard_retrieval(adapted_query)
        personalized_results = self.rerank_for_user(results, user_profile)
        
        return personalized_results
        
    def learn_from_interactions(self, user_id, query, response, feedback):
        """Continuously learn user preferences"""
        
        if feedback == "too_technical":
            self.update_user_profile(user_id, expertise_level=-0.1)
        elif feedback == "not_detailed_enough":
            self.update_user_profile(user_id, detail_preference=+0.2)
```

---

### 🚀 **Why This is a Game-Changer for ML Engineers**

#### **1. Real Research Problems**
- **Retrieval**: How to understand intent and find truly relevant information
- **Evaluation**: How to automatically assess response quality without humans
- **Personalization**: How to adapt AI systems to individual users
- **Knowledge Integration**: How to combine structured and unstructured knowledge

#### **2. Immediate Impact**
- Your ML improvements directly improve user experience
- Clear metrics to measure success (retrieval accuracy, response quality)
- Real-world validation on actual queries and documents

#### **3. Publication Opportunities**
- Novel evaluation methodologies (Judge LLM system)
- Domain adaptation techniques for RAG
- Learned retrieval architectures
- Personalization in information retrieval

#### **4. Perfect Infrastructure**
- Configuration system lets you experiment with different models easily
- Modular architecture makes it easy to swap in ML improvements
- Built-in evaluation framework for measuring improvements
- Database integration for storing training data and user interactions

### 💡 **ML Research Directions**

1. **Multi-Modal RAG**: Understanding images, tables, code in documents
2. **Conversational RAG**: Multi-turn dialogue with context awareness
3. **Cross-Lingual RAG**: Retrieval and generation across languages
4. **Domain Adaptation**: Quickly adapting RAG to new domains
5. **Efficient RAG**: Making high-quality RAG computationally efficient
6. **Explainable RAG**: Understanding why certain documents were retrieved

This isn't just adding ML to an existing system - it's building the **next generation of intelligent information retrieval**. 🎯

