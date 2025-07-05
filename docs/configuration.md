# Configuration Guide

The RAG Engine uses JSON or YAML configuration files to define all aspects of your RAG pipeline. This guide covers all configuration options and provides examples for different use cases.

## Configuration Schema

### Root Structure

```json
{
  "documents": [...],      // Document sources
  "chunking": {...},       // Text chunking strategy
  "embedding": {...},      // Embedding model configuration
  "vectorstore": {...},    // Vector database settings
  "retrieval": {...},      // Retrieval method and parameters
  "llm": {...},           // Language model configuration
  "prompting": {...},     // Prompt templates and settings
  "output": {...}         // Output formatting
}
```

## Document Configuration

Configure document sources and loading strategies.

### Basic Document Loading

```json
{
  "documents": [
    {
      "path": "/path/to/document.txt",
      "type": "text"
    },
    {
      "path": "/path/to/document.pdf", 
      "type": "pdf"
    },
    {
      "path": "/path/to/documents/",
      "type": "directory"
    }
  ]
}
```

### Supported Document Types

| Type | Extensions | Description |
|------|------------|-------------|
| `text` | .txt, .md | Plain text files |
| `pdf` | .pdf | PDF documents |
| `docx` | .docx | Microsoft Word documents |
| `html` | .html, .htm | HTML web pages |
| `directory` | - | Load all files from directory |

### Advanced Document Configuration

```json
{
  "documents": [
    {
      "path": "/data/docs/",
      "type": "directory",
      "recursive": true,
      "patterns": ["*.pdf", "*.txt"],
      "exclude_patterns": ["*temp*"],
      "metadata": {
        "source": "knowledge_base",
        "category": "technical"
      }
    }
  ]
}
```

## Chunking Configuration

Control how documents are split into chunks for processing.

### Fixed Size Chunking

```json
{
  "chunking": {
    "method": "fixed_size",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_tokens": 100,
    "overlap": 50
  }
}
```

### Sentence-Based Chunking

```json
{
  "chunking": {
    "method": "sentence",
    "sentences_per_chunk": 5,
    "overlap_sentences": 1,
    "max_tokens": 200,
    "overlap": 20
  }
}
```

### Token-Based Chunking

```json
{
  "chunking": {
    "method": "token",
    "max_tokens": 150,
    "overlap_tokens": 15,
    "overlap": 15
  }
}
```

### Chunking Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Chunking strategy: `fixed_size`, `sentence`, `token` | `fixed_size` |
| `chunk_size` | Characters per chunk (fixed_size) | 500 |
| `chunk_overlap` | Character overlap between chunks | 50 |
| `max_tokens` | Maximum tokens per chunk | 100 |
| `overlap` | General overlap setting | 50 |

## Embedding Configuration

Configure text embedding models for vector representation.

### Hugging Face Embeddings

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": true
  }
}
```

### OpenAI Embeddings

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "api_key": "${OPENAI_API_KEY}",
    "organization": "${OPENAI_ORG_ID}"
  }
}
```

### Local Embeddings

```json
{
  "embedding": {
    "provider": "local",
    "model_path": "/path/to/local/model",
    "device": "cuda",
    "batch_size": 32
  }
}
```

### Popular Embedding Models

| Provider | Model | Dimensions | Use Case |
|----------|--------|------------|----------|
| Hugging Face | `all-MiniLM-L6-v2` | 384 | General purpose, fast |
| Hugging Face | `all-mpnet-base-v2` | 768 | High quality, slower |
| OpenAI | `text-embedding-ada-002` | 1536 | High quality, API-based |
| Local | Custom BERT models | Varies | Domain-specific |

## Vector Store Configuration

Configure vector databases for storing and retrieving embeddings.

### ChromaDB (Default)

```json
{
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "./vector_db",
    "collection_name": "documents",
    "distance_function": "cosine"
  }
}
```

### FAISS

```json
{
  "vectorstore": {
    "provider": "faiss",
    "index_type": "IndexFlatIP",
    "persist_directory": "./faiss_index",
    "normalize_vectors": true
  }
}
```

### Pinecone

```json
{
  "vectorstore": {
    "provider": "pinecone",
    "api_key": "${PINECONE_API_KEY}",
    "environment": "us-west1-gcp",
    "index_name": "rag-engine",
    "dimension": 384
  }
}
```

### Vector Store Comparison

| Provider | Best For | Pros | Cons |
|----------|----------|------|------|
| ChromaDB | Development, small-medium datasets | Easy setup, persistent | Limited scaling |
| FAISS | High performance, large datasets | Very fast, CPU/GPU support | No built-in persistence |
| Pinecone | Production, cloud deployment | Managed service, scales well | Cost, API dependency |

## Retrieval Configuration

Configure document retrieval methods and parameters.

### Similarity Search

```json
{
  "retrieval": {
    "method": "similarity",
    "top_k": 5,
    "score_threshold": 0.7,
    "include_metadata": true
  }
}
```

### BM25 Search

```json
{
  "retrieval": {
    "method": "bm25",
    "top_k": 10,
    "k1": 1.2,
    "b": 0.75
  }
}
```

### Hybrid Search

```json
{
  "retrieval": {
    "method": "hybrid",
    "top_k": 10,
    "semantic_weight": 0.7,
    "bm25_weight": 0.3,
    "rerank": true,
    "rerank_top_k": 20
  }
}
```

### MMR (Maximal Marginal Relevance)

```json
{
  "retrieval": {
    "method": "mmr",
    "top_k": 5,
    "lambda_mult": 0.5,
    "fetch_k": 20
  }
}
```

## LLM Configuration

Configure language models for response generation.

### OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

### Anthropic Claude

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "api_key": "${ANTHROPIC_API_KEY}",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

### Local Models (Ollama)

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7
  }
}
```

### Local Hugging Face Models

```json
{
  "llm": {
    "provider": "local",
    "model_path": "/path/to/model",
    "tokenizer_path": "/path/to/tokenizer",
    "device": "cuda",
    "max_length": 2048,
    "temperature": 0.7
  }
}
```

## Prompting Configuration

Configure prompt templates and system instructions. The RAG Engine now supports both legacy and enhanced prompting systems with multiple prompter types and template management.

### Enhanced Prompting System

The enhanced prompting system supports multiple prompter types with advanced features:

#### RAG Prompter (Recommended)
```json
{
  "prompting": {
    "type": "rag",
    "template_path": "./templates/rag_template.txt",
    "system_prompt": "You are a helpful AI assistant.",
    "context_window": 3000,
    "citation_format": "numbered",
    "context_optimization": {
      "relevance_filtering": true,
      "diversity_enhancement": true,
      "redundancy_removal": true,
      "length_balancing": true
    }
  }
}
```

#### Conversational Prompter
```json
{
  "prompting": {
    "type": "conversational",
    "template_path": "./templates/chat_template.txt",
    "system_prompt": "You are a friendly AI assistant.",
    "memory_length": 10,
    "context_compression": true,
    "persona": "helpful_assistant",
    "conversation_template": "chat_template.txt"
  }
}
```

#### Code Explanation Prompter
```json
{
  "prompting": {
    "type": "code_explanation",
    "template_path": "./templates/code_template.txt",
    "system_prompt": "You are an expert code assistant.",
    "language": "python",
    "include_syntax_highlighting": true,
    "add_comments": true,
    "explain_logic": true
  }
}
```

#### Debugging Prompter
```json
{
  "prompting": {
    "type": "debugging",
    "template_path": "./templates/debug_template.txt",
    "system_prompt": "You are a debugging expert.",
    "debug_level": "detailed",
    "include_stack_trace": true,
    "suggest_fixes": true
  }
}
```

#### Chain of Thought Prompter
```json
{
  "prompting": {
    "type": "chain_of_thought",
    "template_path": "./templates/chain_of_thought_template.txt",
    "system_prompt": "Think step by step to solve problems.",
    "reasoning_steps": 3,
    "explicit_reasoning": true,
    "step_separator": "\nStep {n}:",
    "conclusion_prompt": "Therefore:"
  }
}
```

### Legacy Prompting (Backward Compatible)

### Default Prompting

```json
{
  "prompting": {
    "template": "default",
    "system_prompt": "You are a helpful assistant that answers questions based on the provided context.",
    "context_template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    "max_context_length": 2000
  }
}
```

### Conversational Prompting

```json
{
  "prompting": {
    "template": "conversational",
    "system_prompt": "You are a friendly AI assistant. Use the provided context to answer questions naturally.",
    "include_chat_history": true,
    "max_history_turns": 5
  }
}
```

### Q&A Prompting

```json
{
  "prompting": {
    "template": "qa",
    "system_prompt": "Answer the question based only on the provided context. If the answer is not in the context, say 'I don't know'.",
    "strict_context": true,
    "require_citations": true
  }
}
```

### Template Management

#### Custom Templates
Templates are stored in the `templates/` directory and can be customized:

- `rag_template.txt` - Standard RAG prompts
- `chat_template.txt` - Conversational prompts  
- `chain_of_thought_template.txt` - Reasoning prompts
- `code_template.txt` - Code explanation prompts
- `debug_template.txt` - Debugging prompts

#### Template Variables
Templates support variable substitution:
- `{query}` - User question
- `{context}` - Retrieved documents
- `{conversation_history}` - Previous conversation
- `{metadata}` - Document metadata
- `{custom_vars}` - Custom template variables

#### Advanced Features
```json
{
  "prompting": {
    "type": "rag",
    "advanced_features": {
      "dynamic_template_selection": true,
      "multi_language_support": true,
      "context_compression": true,
      "citation_enhancement": true,
      "quality_optimization": true
    }
  }
}
```

## Output Configuration

Configure response formatting and output options.

### Basic Output

```json
{
  "output": {
    "method": "direct",
    "format": "json",
    "include_sources": true,
    "include_scores": false
  }
}
```

### Structured Output

```json
{
  "output": {
    "method": "structured",
    "format": "json",
    "schema": {
      "answer": "string",
      "confidence": "number",
      "sources": "array",
      "reasoning": "string"
    }
  }
}
```

## Environment Variables

Use environment variables for sensitive configuration values.

### Supported Variables

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key

# Organization IDs
OPENAI_ORG_ID=your_openai_org
PINECONE_ENVIRONMENT=us-west1-gcp

# Database URLs
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://localhost:6379

# Custom Settings
RAG_DEBUG=true
RAG_LOG_LEVEL=INFO
```

### Using in Configuration

```json
{
  "llm": {
    "provider": "openai",
    "api_key": "${OPENAI_API_KEY}",
    "organization": "${OPENAI_ORG_ID}"
  },
  "debug": "${RAG_DEBUG:false}"
}
```

## Complete Configuration Examples

### Development Configuration

```json
{
  "documents": [
    {"path": "./docs", "type": "directory"}
  ],
  "chunking": {
    "method": "fixed_size",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_tokens": 100,
    "overlap": 50
  },
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "vectorstore": {
    "provider": "chroma",
    "persist_directory": "./vector_db"
  },
  "retrieval": {
    "method": "similarity",
    "top_k": 5
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.7
  },
  "prompting": {
    "template": "default",
    "system_prompt": "You are a helpful assistant."
  },
  "output": {
    "method": "direct"
  }
}
```

### Production Configuration

```json
{
  "documents": [
    {
      "path": "/data/knowledge_base",
      "type": "directory",
      "recursive": true,
      "patterns": ["*.pdf", "*.docx", "*.txt"]
    }
  ],
  "chunking": {
    "method": "sentence",
    "sentences_per_chunk": 5,
    "overlap_sentences": 1,
    "max_tokens": 200,
    "overlap": 20
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "api_key": "${OPENAI_API_KEY}"
  },
  "vectorstore": {
    "provider": "pinecone",
    "api_key": "${PINECONE_API_KEY}",
    "environment": "us-west1-gcp",
    "index_name": "production-rag",
    "dimension": 1536
  },
  "retrieval": {
    "method": "hybrid",
    "top_k": 10,
    "semantic_weight": 0.7,
    "bm25_weight": 0.3,
    "rerank": true
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.7,
    "max_tokens": 1500
  },
  "prompting": {
    "template": "qa",
    "system_prompt": "You are an expert AI assistant. Provide accurate answers based on the context.",
    "strict_context": true,
    "require_citations": true
  },
  "output": {
    "method": "structured",
    "format": "json",
    "include_sources": true,
    "include_scores": true
  }
}
```

## Configuration Validation

The RAG Engine validates configurations against a Pydantic schema. Common validation errors:

### Missing Required Fields

```bash
Error: Field required [type=missing, input_value=..., input_type=dict]
```

**Solution**: Add all required fields to your configuration.

### Invalid Values

```bash
Error: Input should be 'similarity', 'bm25', 'hybrid' or 'mmr'
```

**Solution**: Use only supported values as documented.

### Environment Variable Not Found

```bash
Error: Environment variable 'OPENAI_API_KEY' not found
```

**Solution**: Set the required environment variable or provide the value directly.

## Best Practices

1. **Use Environment Variables**: Keep sensitive data out of config files
2. **Start Simple**: Begin with default settings and optimize as needed
3. **Version Control**: Track configuration changes with your code
4. **Validate Early**: Test configurations with small datasets first
5. **Document Changes**: Comment complex configuration decisions
6. **Monitor Performance**: Adjust parameters based on actual usage
