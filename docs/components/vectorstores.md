# Vector Stores

Vector stores are databases optimized for storing and querying high-dimensional embeddings. They enable fast similarity search, which is crucial for efficient document retrieval in RAG systems.

## Available Vector Stores

### ChromaDB
Open-source, lightweight vector database with excellent Python integration.

**Configuration:**
```json
{
  "vectorstore": {
    "type": "chroma",
    "config": {
      "persist_directory": "./chroma_db",
      "collection_name": "documents",
      "distance_metric": "cosine",
      "embedding_function": "auto"
    }
  }
}
```

**Features:**
- Local and client-server modes
- Automatic persistence
- Metadata filtering
- Built-in embedding functions

### FAISS
Facebook AI Similarity Search - high-performance vector search library.

**Configuration:**
```json
{
  "vectorstore": {
    "type": "faiss",
    "config": {
      "index_type": "IndexFlatIP",
      "dimension": 384,
      "metric_type": "INNER_PRODUCT",
      "nlist": 100,
      "nprobe": 10
    }
  }
}
```

**Index Types:**
- `IndexFlatIP`: Exact search with inner product
- `IndexFlatL2`: Exact search with L2 distance
- `IndexIVFFlat`: Inverted file with exact post-verification
- `IndexIVFPQ`: Inverted file with product quantization

### Pinecone
Managed vector database service with enterprise features.

**Configuration:**
```json
{
  "vectorstore": {
    "type": "pinecone",
    "config": {
      "api_key": "${PINECONE_API_KEY}",
      "environment": "us-west1-gcp",
      "index_name": "rag-documents",
      "dimension": 1536,
      "metric": "cosine",
      "pod_type": "p1.x1"
    }
  }
}
```

**Features:**
- Fully managed service
- Auto-scaling
- Real-time updates
- Advanced filtering

### Weaviate
Open-source vector database with GraphQL API.

**Configuration:**
```json
{
  "vectorstore": {
    "type": "weaviate",
    "config": {
      "url": "http://localhost:8080",
      "api_key": "${WEAVIATE_API_KEY}",
      "class_name": "Document",
      "vector_index_type": "hnsw",
      "distance_metric": "cosine"
    }
  }
}
```

### Qdrant
High-performance vector search engine with advanced filtering.

**Configuration:**
```json
{
  "vectorstore": {
    "type": "qdrant",
    "config": {
      "host": "localhost",
      "port": 6333,
      "collection_name": "documents",
      "vector_size": 384,
      "distance": "Cosine",
      "on_disk": true
    }
  }
}
```

## Usage Examples

### Basic Operations
```python
from rag_engine.core.vectorstore import ChromaVectorStore

# Initialize vector store
vectorstore = ChromaVectorStore({
    "persist_directory": "./my_vectorstore",
    "collection_name": "documents"
})

# Add documents
documents = [
    {
        "content": "Python is a programming language",
        "embedding": [0.1, 0.2, 0.3, ...],
        "metadata": {"source": "doc1.txt", "type": "text"}
    },
    {
        "content": "Machine learning is a subset of AI",
        "embedding": [0.4, 0.5, 0.6, ...],
        "metadata": {"source": "doc2.txt", "type": "text"}
    }
]

vectorstore.add_documents(documents)

# Search similar documents
query_embedding = [0.15, 0.25, 0.35, ...]
results = vectorstore.similarity_search(
    query_embedding, 
    k=5, 
    score_threshold=0.7
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Metadata: {result['metadata']}")
```

### Advanced Filtering
```python
# Search with metadata filters
results = vectorstore.similarity_search(
    query_embedding,
    k=10,
    filter={
        "source": {"$in": ["doc1.txt", "doc2.txt"]},
        "type": "text",
        "date": {"$gte": "2024-01-01"}
    }
)

# Boolean filters
results = vectorstore.similarity_search(
    query_embedding,
    k=5,
    filter={
        "$and": [
            {"category": "science"},
            {"$or": [
                {"author": "Smith"},
                {"author": "Johnson"}
            ]}
        ]
    }
)
```

### Batch Operations
```python
# Batch add documents
batch_documents = [doc1, doc2, doc3, ...]
vectorstore.add_documents_batch(batch_documents, batch_size=100)

# Batch search
query_embeddings = [emb1, emb2, emb3, ...]
batch_results = vectorstore.similarity_search_batch(query_embeddings)
```

## Creating Custom Vector Stores

Implement the `BaseVectorStore` interface:

```python
from rag_engine.core.base import BaseVectorStore
from typing import List, Dict, Any, Optional
import numpy as np

class CustomVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dimension = config.get("dimension")
        self.collection_name = config.get("collection_name", "default")
        # Initialize your vector store
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        document_ids = []
        for doc in documents:
            doc_id = self._add_single_document(doc)
            document_ids.append(doc_id)
        return document_ids
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filter: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        # Implement similarity search logic
        return results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs."""
        # Implement deletion logic
        return True
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_documents": self._count_documents(),
            "dimension": self.dimension,
            "collection_name": self.collection_name
        }

# Register the vector store
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("vectorstore", "custom", CustomVectorStore)
```

## Performance Optimization

### Index Configuration
```json
{
  "vectorstore": {
    "type": "faiss",
    "config": {
      "index_type": "IndexIVFPQ",
      "nlist": 1024,
      "m": 8,
      "nbits": 8,
      "nprobe": 64,
      "train_size": 10000
    }
  }
}
```

### Memory Management
```json
{
  "vectorstore": {
    "config": {
      "memory_map": true,
      "cache_size": "1GB",
      "batch_size": 1000,
      "parallel_workers": 4
    }
  }
}
```

### Search Optimization
```json
{
  "vectorstore": {
    "config": {
      "search_params": {
        "ef": 200,
        "max_connections": 64,
        "ef_construction": 400
      },
      "enable_search_cache": true,
      "cache_ttl": 3600
    }
  }
}
```

## Vector Store Comparison

| Feature | ChromaDB | FAISS | Pinecone | Weaviate | Qdrant |
|---------|----------|-------|----------|-----------|---------|
| **Deployment** | Local/Server | Local | Cloud | Local/Cloud | Local/Cloud |
| **Scaling** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Filtering** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | Free | Free | Paid | Free/Paid | Free/Paid |

## Advanced Features

### Hybrid Search
Combine vector and keyword search:

```json
{
  "vectorstore": {
    "config": {
      "hybrid_search": {
        "enabled": true,
        "keyword_weight": 0.3,
        "vector_weight": 0.7,
        "fusion_method": "rrf"
      }
    }
  }
}
```

### Multi-vector Storage
Store multiple embeddings per document:

```json
{
  "vectorstore": {
    "config": {
      "multi_vector": {
        "enabled": true,
        "vector_types": ["dense", "sparse", "summary"],
        "aggregation_method": "weighted_average"
      }
    }
  }
}
```

### Hierarchical Storage
Organize vectors in hierarchies:

```json
{
  "vectorstore": {
    "config": {
      "hierarchical": {
        "levels": ["document", "section", "paragraph"],
        "navigation_enabled": true,
        "parent_child_relations": true
      }
    }
  }
}
```

## Monitoring and Maintenance

### Performance Monitoring
```json
{
  "vectorstore": {
    "config": {
      "monitoring": {
        "track_query_latency": true,
        "track_memory_usage": true,
        "log_slow_queries": true,
        "metrics_endpoint": "/metrics"
      }
    }
  }
}
```

### Maintenance Tasks
```python
# Index optimization
vectorstore.optimize_index()

# Statistics collection
stats = vectorstore.get_collection_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Index size: {stats['index_size_mb']} MB")

# Health check
health = vectorstore.health_check()
print(f"Status: {health['status']}")
```

### Backup and Recovery
```json
{
  "vectorstore": {
    "config": {
      "backup": {
        "enabled": true,
        "schedule": "0 2 * * *",
        "retention_days": 30,
        "backup_location": "s3://my-backup-bucket"
      }
    }
  }
}
```

## Migration Between Vector Stores

### Export/Import
```python
from rag_engine.core.migration import VectorStoreMigrator

migrator = VectorStoreMigrator()

# Export from ChromaDB
data = migrator.export_vectorstore(
    source_type="chroma",
    source_config=chroma_config
)

# Import to Pinecone
migrator.import_vectorstore(
    target_type="pinecone",
    target_config=pinecone_config,
    data=data
)
```

### Schema Mapping
```json
{
  "migration": {
    "field_mapping": {
      "content": "text",
      "metadata.source": "source_file",
      "metadata.timestamp": "created_at"
    },
    "vector_dimension_mapping": {
      "source": 384,
      "target": 1536,
      "transform_method": "pad_or_truncate"
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Memory Issues**
```json
{
  "vectorstore": {
    "config": {
      "memory_optimization": {
        "use_memory_mapping": true,
        "batch_size": 500,
        "clear_cache_interval": 1000
      }
    }
  }
}
```

**2. Slow Queries**
```json
{
  "vectorstore": {
    "config": {
      "performance_tuning": {
        "index_type": "approximate",
        "search_parallelism": 4,
        "enable_query_cache": true
      }
    }
  }
}
```

**3. Index Corruption**
```python
# Rebuild index
vectorstore.rebuild_index()

# Verify integrity
vectorstore.verify_index_integrity()

# Repair if needed
vectorstore.repair_index()
```

**4. Connection Issues**
```json
{
  "vectorstore": {
    "config": {
      "connection": {
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "timeout": 30,
        "health_check_interval": 60
      }
    }
  }
}
```

## Dependencies

Install required packages based on vector store type:

```bash
# ChromaDB
pip install chromadb

# FAISS
pip install faiss-cpu  # or faiss-gpu

# Pinecone
pip install pinecone-client

# Weaviate
pip install weaviate-client

# Qdrant
pip install qdrant-client
```
