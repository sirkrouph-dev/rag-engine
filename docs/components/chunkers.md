# Text Chunkers

Chunkers split large documents into smaller, manageable pieces for embedding and retrieval. The choice of chunking strategy significantly impacts RAG performance.

## Available Chunkers

### FixedSizeChunker
Splits text into chunks of fixed character or token length.

**Configuration:**
```json
{
  "chunker": {
    "type": "fixed_size",
    "config": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "separator": "\n\n",
      "keep_separator": true
    }
  }
}
```

**Parameters:**
- `chunk_size`: Maximum characters per chunk
- `chunk_overlap`: Overlap between consecutive chunks
- `separator`: Preferred split points
- `keep_separator`: Preserve separators in chunks

### SentenceChunker
Splits text at sentence boundaries for semantic coherence.

**Configuration:**
```json
{
  "chunker": {
    "type": "sentence",
    "config": {
      "max_sentences": 5,
      "min_chunk_size": 100,
      "max_chunk_size": 1500,
      "overlap_sentences": 1
    }
  }
}
```

**Features:**
- Preserves sentence integrity
- Configurable sentence count per chunk
- Size constraints with sentence boundaries
- Smart overlap handling

### TokenChunker
Splits based on token count using tokenizer-aware boundaries.

**Configuration:**
```json
{
  "chunker": {
    "type": "token",
    "config": {
      "max_tokens": 512,
      "overlap_tokens": 50,
      "tokenizer": "gpt-3.5-turbo",
      "preserve_words": true
    }
  }
}
```

**Tokenizer Support:**
- OpenAI models (gpt-3.5-turbo, gpt-4)
- Hugging Face tokenizers
- Custom tokenizers

### SemanticChunker
Advanced chunking using embedding similarity.

**Configuration:**
```json
{
  "chunker": {
    "type": "semantic",
    "config": {
      "similarity_threshold": 0.8,
      "min_chunk_size": 200,
      "max_chunk_size": 1000,
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

**Features:**
- Semantic boundary detection
- Embedding-based similarity scoring
- Dynamic chunk sizing
- Content-aware splitting

## Usage Examples

### Basic Chunking
```python
from rag_engine.core.chunker import FixedSizeChunker

chunker = FixedSizeChunker({
    "chunk_size": 1000,
    "chunk_overlap": 200
})

text = "Your long document text here..."
chunks = chunker.chunk(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:100]}...")
```

### Document Processing
```python
from rag_engine.core.orchestration import ComponentRegistry

registry = ComponentRegistry()
chunker = registry.get_component("chunker", "sentence", config)

documents = [
    {"content": "Document 1 text...", "metadata": {"source": "doc1.txt"}},
    {"content": "Document 2 text...", "metadata": {"source": "doc2.txt"}}
]

chunked_docs = []
for doc in documents:
    chunks = chunker.chunk(doc["content"])
    for i, chunk in enumerate(chunks):
        chunked_docs.append({
            "content": chunk,
            "metadata": {
                **doc["metadata"],
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })
```

### Adaptive Chunking
```python
chunker_config = {
    "type": "adaptive",
    "config": {
        "min_size": 100,
        "max_size": 1500,
        "strategies": ["sentence", "paragraph", "section"],
        "prefer_semantic": True
    }
}
```

## Chunking Strategies

### 1. Content-Aware Chunking
Different content types require different strategies:

```json
{
  "chunker": {
    "type": "content_aware",
    "config": {
      "strategies": {
        "code": {"type": "function", "preserve_structure": true},
        "markdown": {"type": "section", "respect_headers": true},
        "plain_text": {"type": "sentence", "max_sentences": 5}
      }
    }
  }
}
```

### 2. Hierarchical Chunking
Create chunks at multiple levels:

```json
{
  "chunker": {
    "type": "hierarchical",
    "config": {
      "levels": [
        {"type": "section", "max_size": 5000},
        {"type": "paragraph", "max_size": 1500},
        {"type": "sentence", "max_size": 500}
      ]
    }
  }
}
```

### 3. Overlap Strategies
Control how chunks overlap for better context:

```json
{
  "chunker": {
    "config": {
      "overlap_strategy": "sliding_window",
      "overlap_percentage": 0.2,
      "overlap_units": "sentences"
    }
  }
}
```

## Creating Custom Chunkers

Implement the `BaseChunker` interface:

```python
from rag_engine.core.base import BaseChunker
from typing import List, Dict, Any

class CustomChunker(BaseChunker):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 1000)
        self.overlap = config.get("chunk_overlap", 200)
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        # Implement your chunking logic
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents."""
        chunked_docs = []
        for doc in documents:
            chunks = self.chunk(doc["content"])
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "content": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_id": i,
                        "source_doc": doc.get("id", "unknown")
                    }
                })
        return chunked_docs

# Register the chunker
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("chunker", "custom", CustomChunker)
```

## Configuration Guidelines

### Choosing Chunk Size
The optimal chunk size depends on your use case:

| Use Case | Recommended Size | Strategy |
|----------|------------------|-----------|
| **Q&A** | 500-1000 chars | Sentence-based |
| **Summarization** | 1000-2000 chars | Paragraph-based |
| **Code Search** | Function/class level | Structure-aware |
| **Legal Documents** | Section-based | Hierarchical |

### Performance Tuning
```json
{
  "chunker": {
    "config": {
      "parallel_processing": true,
      "batch_size": 100,
      "cache_results": true,
      "memory_efficient": true
    }
  }
}
```

### Quality Optimization
```json
{
  "chunker": {
    "config": {
      "preserve_context": true,
      "avoid_orphans": true,
      "min_meaningful_size": 50,
      "quality_threshold": 0.7
    }
  }
}
```

## Advanced Features

### Metadata Preservation
```python
chunker_config = {
    "preserve_metadata": True,
    "metadata_fields": ["title", "author", "section"],
    "add_chunk_metadata": {
        "chunk_index": True,
        "total_chunks": True,
        "overlap_info": True
    }
}
```

### Multi-language Support
```json
{
  "chunker": {
    "config": {
      "language": "auto",
      "language_specific_rules": {
        "chinese": {"sentence_segmentation": "jieba"},
        "japanese": {"sentence_segmentation": "mecab"},
        "arabic": {"rtl_support": true}
      }
    }
  }
}
```

### Content Type Detection
```python
chunker_config = {
    "auto_detect_content": True,
    "content_handlers": {
        "code": "CodeChunker",
        "table": "TableChunker", 
        "list": "ListChunker",
        "citation": "CitationChunker"
    }
}
```

## Performance Considerations

### Memory Usage
- Process large documents in batches
- Use streaming for very large files
- Configure appropriate chunk sizes

### Processing Speed
- Enable parallel processing for multiple documents
- Cache chunking results for repeated use
- Use appropriate tokenizers for your content

### Quality vs. Speed
Balance chunk quality with processing speed:

```json
{
  "chunker": {
    "config": {
      "quality_mode": "balanced",
      "max_processing_time": 30,
      "fallback_chunker": "fixed_size"
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Chunks Too Small/Large**
```json
{
  "chunker": {
    "config": {
      "adaptive_sizing": true,
      "target_size_range": [200, 1500],
      "size_tolerance": 0.2
    }
  }
}
```

**2. Lost Context at Boundaries**
```json
{
  "chunker": {
    "config": {
      "context_preservation": "smart_overlap",
      "boundary_detection": "semantic",
      "overlap_quality_check": true
    }
  }
}
```

**3. Performance Issues**
```json
{
  "chunker": {
    "config": {
      "optimize_for": "speed",
      "use_approximations": true,
      "batch_processing": true
    }
  }
}
```
