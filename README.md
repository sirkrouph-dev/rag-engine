# RAG Engine

A powerful, modular framework for building advanced Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code. Combining state-of-the-art retrieval algorithms with multiple vector databases and LLM providers.

> **Note:** This project is currently a Work in Progress (WIP). Core functionality including document processing, embedding, vector storage, and retrieval strategies are implemented, but the API server, UI, and evaluation tools are still under development. Documentation and examples are being actively updated.
>
> _Built with ❤️ and GitHub Copilot - Transforming vision into code._

## 🗺️ Project Roadmap

### Current Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| Core Architecture | ✅ Implemented | Base interfaces, module structure, configuration schema |
| Document Loading | 🟡 In Progress | Basic PDF, TXT, & HTML loading framework |
| Chunking | 🟡 In Progress | Fixed size & recursive chunking strategies |
| Embedding | 🟡 In Progress | OpenAI embeddings integration, framework for alternatives |
| Vector Stores | 🟡 In Progress | FAISS & ChromaDB basic integration |
| Retrieval | 🔴 Planned | Simple similarity implemented, advanced retrieval strategies in development |
| LLM Integration | 🟡 In Progress | OpenAI API integration complete, alternatives in development |
| Config System | ✅ Implemented | JSON/YAML parsing with Pydantic validation |
| CLI | 🟡 In Progress | Basic commands scaffolded, functionality being connected |
| REST API | 🔴 Planned | FastAPI structure defined, endpoints in development |
| Web UI | 🔴 Planned | Not yet implemented |
| Plugin System | 🔴 Planned | Interface defined, implementation pending |
| Evaluation Tools | 🔴 Planned | Not yet implemented |
| Metadata Extraction | 🔴 Planned | Not yet implemented |
| Query Processing | 🔴 Planned | Not yet implemented |
| Tools Integration | 🔴 Planned | Architecture design phase, implementation not started |
| Tool Registry | 🔴 Planned | Not yet implemented |
| Data Enrichment Tools | 🔴 Planned | Not yet implemented |
| Content Processing Tools | 🔴 Planned | Not yet implemented |
| Advanced Reasoning Tools | 🔴 Planned | Not yet implemented |

### Development Timeline

- **Q2 2025** (Current)
  - Complete basic document loading for all supported formats
  - Finish core chunking strategies implementation
  - Implement basic retrieval functionality
  - Release initial CLI functionality for build & chat

- **Q3 2025**
  - Complete advanced retrieval strategies
  - Implement FastAPI server with basic endpoints
  - Add metadata extraction functionality
  - Integrate additional vector stores and embedders
  - Begin tool integration framework development

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

> **Note:** The project is under active development, and some features shown in these examples are still in implementation. Please check the [Project Roadmap](#-project-roadmap) section to see what's currently available.

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

