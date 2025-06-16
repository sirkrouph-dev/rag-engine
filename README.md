# RAG Engine

A powerful, modular framework for building advanced Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code. Combining state-of-the-art retrieval algorithms with multiple vector databases and LLM providers.

> **Status: Early Development** 🚧 
> 
> This project is in early development with core components implemented but pipeline integration still in progress. The extensive documentation represents the target vision, not current functionality. See the roadmap below for implementation status.
>
> _Built with ❤️ and GitHub Copilot - Learning and building in public._

## 🗺️ Project Roadmap

### Current Implementation Status

| Component | Status | Implementation Details |
|-----------|--------|----------------------|
| **Core Architecture** | ✅ **Complete** | Base interfaces and module structure implemented in `rag_engine/core/base.py` |
| **Configuration System** | ✅ **Complete** | Pydantic schemas in `rag_engine/config/schema.py`, YAML/JSON parsing |
| **Document Loading** | ✅ **Implemented** | PDF, TXT, DOCX, HTML loaders in `rag_engine/core/loader.py` |
| **Text Chunking** | ✅ **Implemented** | Fixed-size, recursive chunking strategies in `rag_engine/core/chunker.py` |
| **Embedding** | ✅ **Implemented** | OpenAI, Ollama, HuggingFace providers in `rag_engine/core/embedder.py` |
| **Vector Stores** | ✅ **Implemented** | FAISS, ChromaDB, Pinecone, Qdrant in `rag_engine/core/vectorstore.py` |
| **Retrieval** | ✅ **Implemented** | Multiple strategies (similarity, MMR, hybrid) in `rag_engine/core/retriever.py` |
| **LLM Integration** | ✅ **Implemented** | OpenAI, Gemini, Local models in `rag_engine/core/llm.py` |
| **Tools System** | ✅ **Implemented** | Web search, calculator, file ops in `rag_engine/core/tools.py` and `rag_engine/plugins/tools.py` |
| **Reasoning Engine** | ✅ **Implemented** | Chain-of-thought, tree-of-thought in `rag_engine/core/reasoning.py` |
| **CLI Framework** | 🟡 **Partially Working** | Commands defined in `rag_engine/__main__.py` but not fully connected to pipeline |
| **Pipeline Integration** | 🔴 **Not Working** | Core pipeline exists but component integration incomplete |
| **REST API** | � **Placeholder Only** | Basic FastAPI structure in `rag_engine/interfaces/api.py` |
| **Web UI** | 🔴 **Not Started** | Empty file at `rag_engine/interfaces/ui.py` |
| **Database Integration** | 🔴 **Planned** | Chat history, knowledge graphs not implemented |
| **Testing Framework** | 🔴 **Not Started** | No tests exist yet |

### What Actually Works Right Now
- 🟡 **Individual Components**: All core components can be imported and used independently
- 🟡 **Configuration Loading**: YAML/JSON configs are parsed and validated
- 🔴 **End-to-End Pipeline**: Full RAG workflow not yet functional
- 🔴 **CLI Commands**: `python -m rag_engine` exists but commands don't execute pipeline
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
- **Modular Architecture**: Basic interface design for swappable components
- **Configuration Validation**: Pydantic-based schema validation for YAML/JSON configs
- **Basic Document Loading**: Framework for TXT, PDF loading (implementation in progress)
- **Basic Chunking**: Fixed-size and recursive chunking strategies (implementation in progress)
- **OpenAI Embeddings**: Integration with OpenAI embedding API
- **Simple Vector Stores**: Initial FAISS and ChromaDB integration
- **Basic CLI**: Command structure with Typer (functionality being connected)

### Coming Soon

- **Multiple Document Types**: Complete support for TXT, PDF, DOCX, and HTML documents
- **Advanced Chunking**: Sentence-based and semantic chunking strategies
- **Flexible Embeddings**: Google Gemini/Vertex, SentenceTransformers integration
- **Comprehensive Retrieval Strategies**:
  - Basic: Simple similarity, threshold filtering, MMR, hybrid search
  - Advanced: Contextual compression, multi-query expansion, hierarchical retrieval
- **Additional Vector Stores**:
  - Local: PostgreSQL (pgVector)
  - Cloud: Pinecone, Qdrant
- **Flexible LLM Support**:
  - Cloud providers: Google Gemini
  - Local models: Phi-3, Gemma, and any model via Ollama
- **Multi-Interface**: Complete REST API and web UI
- **Plugin System**: Extend with custom components
- **Metadata Extraction**: Document and query-level metadata processing
- **Query Classification**: Intelligent routing and processing

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

> **Important**: The full pipeline is not yet functional. You can explore individual components and configuration, but end-to-end RAG is still in development.

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

### 2. Explore the Configuration System (✅ Working)

Create a config file to see how the system is designed:

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
    "persist_directory": "./vector_store",
    "collection_name": "rag_docs"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.3
  }
}
```

### 3. Test Individual Components (✅ Working)

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

### Current Limitations
- Pipeline integration incomplete
- CLI commands are placeholder
- No working end-to-end examples
- API endpoints not functional
- No tests or validation

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

### Where ML Fits in RAG Engine

As an ML engineer, here are the key areas where your skills are valuable:

#### 1. **Embedding Model Optimization**
```python
# Current: Basic provider integration
# ML Opportunity: Custom embedding fine-tuning
{
  "embedding": {
    "provider": "custom",
    "model_path": "./models/domain_specific_embedder",
    "fine_tuned_for": "legal_documents"
  }
}
```

#### 2. **Retrieval Model Development**
```python
# Current: Vector similarity, MMR, hybrid search
# ML Opportunity: Learned dense retrieval (ColBERT, DPR)
class LearnedDenseRetriever(RetrieverStrategy):
    def __init__(self, query_encoder, doc_encoder):
        self.query_encoder = query_encoder  # Your trained model
        self.doc_encoder = doc_encoder
```

#### 3. **Reranking Models**
```python
# Current: Basic similarity reranking
# ML Opportunity: Cross-encoder rerankers
{
  "retrieval": {
    "strategy": "rerank",
    "reranker": {
      "type": "cross_encoder",
      "model": "ms-marco-MiniLM-L-6-v2",
      "batch_size": 32
    }
  }
}
```

#### 4. **Query Understanding & Classification**
```python
# Current: Basic text classification
# ML Opportunity: Intent classification, entity extraction
class MLQueryProcessor:
    def classify_intent(self, query: str) -> QueryIntent
    def extract_entities(self, query: str) -> List[Entity]
    def expand_query(self, query: str) -> List[str]
```

