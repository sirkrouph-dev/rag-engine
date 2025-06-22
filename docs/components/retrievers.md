# Document Retrievers

Retrievers are responsible for finding the most relevant documents from the vector store based on a query. They implement various search strategies and can combine multiple retrieval methods for better results.

## Available Retrievers

### SimilarityRetriever
Basic similarity search using vector embeddings.

**Configuration:**
```json
{
  "retriever": {
    "type": "similarity",
    "config": {
      "top_k": 5,
      "score_threshold": 0.7,
      "distance_metric": "cosine",
      "normalize_scores": true
    }
  }
}
```

**Features:**
- Pure vector similarity search
- Configurable similarity thresholds
- Multiple distance metrics
- Score normalization

### BM25Retriever
Traditional keyword-based search using BM25 algorithm.

**Configuration:**
```json
{
  "retriever": {
    "type": "bm25",
    "config": {
      "top_k": 10,
      "k1": 1.2,
      "b": 0.75,
      "epsilon": 0.25,
      "use_stemming": true,
      "remove_stopwords": true
    }
  }
}
```

**Parameters:**
- `k1`: Term frequency saturation parameter
- `b`: Field length normalization parameter
- `epsilon`: IDF smoothing parameter

### HybridRetriever
Combines vector similarity and keyword search.

**Configuration:**
```json
{
  "retriever": {
    "type": "hybrid",
    "config": {
      "vector_weight": 0.7,
      "keyword_weight": 0.3,
      "fusion_method": "rrf",
      "rrf_k": 60,
      "top_k": 5
    }
  }
}
```

**Fusion Methods:**
- `rrf`: Reciprocal Rank Fusion
- `weighted_sum`: Weighted score combination
- `convex_combination`: Convex combination of scores

### MMRRetriever
Maximal Marginal Relevance for diverse results.

**Configuration:**
```json
{
  "retriever": {
    "type": "mmr",
    "config": {
      "top_k": 5,
      "fetch_k": 20,
      "lambda_diversity": 0.5,
      "similarity_threshold": 0.6
    }
  }
}
```

**Parameters:**
- `fetch_k`: Initial candidates to fetch
- `lambda_diversity`: Balance between relevance and diversity (0-1)

### MultiQueryRetriever
Generates multiple query variations for better coverage.

**Configuration:**
```json
{
  "retriever": {
    "type": "multi_query",
    "config": {
      "num_queries": 3,
      "query_generator": "llm",
      "aggregation_method": "union",
      "min_score": 0.5
    }
  }
}
```

## Usage Examples

### Basic Retrieval
```python
from rag_engine.core.retriever import SimilarityRetriever

retriever = SimilarityRetriever({
    "top_k": 5,
    "score_threshold": 0.7
})

# Set vector store
retriever.set_vectorstore(vectorstore)

# Retrieve documents
query = "What is machine learning?"
results = retriever.retrieve(query)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print("---")
```

### Hybrid Retrieval
```python
from rag_engine.core.retriever import HybridRetriever

retriever = HybridRetriever({
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "fusion_method": "rrf"
})

retriever.set_vectorstore(vectorstore)
retriever.set_keyword_index(bm25_index)

results = retriever.retrieve("machine learning algorithms", top_k=10)
```

### MMR for Diverse Results
```python
from rag_engine.core.retriever import MMRRetriever

retriever = MMRRetriever({
    "top_k": 5,
    "fetch_k": 20,
    "lambda_diversity": 0.6  # Higher diversity
})

retriever.set_vectorstore(vectorstore)
results = retriever.retrieve("python programming")

# Results will be relevant but diverse
```

### Multi-Query Expansion
```python
from rag_engine.core.retriever import MultiQueryRetriever

retriever = MultiQueryRetriever({
    "num_queries": 3,
    "query_generator": "llm"
})

retriever.set_llm(llm)
retriever.set_vectorstore(vectorstore)

# Automatically generates query variations
results = retriever.retrieve("climate change effects")
```

## Advanced Retrieval Strategies

### Contextual Retrieval
Add context to improve retrieval:

```json
{
  "retriever": {
    "type": "contextual",
    "config": {
      "context_window": 3,
      "include_neighbors": true,
      "context_weight": 0.3
    }
  }
}
```

### Hierarchical Retrieval
Search at multiple document levels:

```json
{
  "retriever": {
    "type": "hierarchical",
    "config": {
      "levels": ["section", "paragraph", "sentence"],
      "level_weights": [0.5, 0.3, 0.2],
      "top_k_per_level": [2, 3, 5]
    }
  }
}
```

### Time-aware Retrieval
Consider document recency:

```json
{
  "retriever": {
    "type": "temporal",
    "config": {
      "time_decay_factor": 0.1,
      "recency_weight": 0.2,
      "time_field": "created_at"
    }
  }
}
```

## Creating Custom Retrievers

Implement the `BaseRetriever` interface:

```python
from rag_engine.core.base import BaseRetriever
from typing import List, Dict, Any, Optional

class CustomRetriever(BaseRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.top_k = config.get("top_k", 5)
        self.custom_param = config.get("custom_param", "default")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query."""
        if top_k is None:
            top_k = self.top_k
        
        # Implement your retrieval logic
        results = self._custom_retrieval(query, top_k, filter)
        
        # Ensure required fields
        for result in results:
            if "score" not in result:
                result["score"] = 1.0
            if "metadata" not in result:
                result["metadata"] = {}
        
        return results
    
    def _custom_retrieval(self, query: str, top_k: int, filter: Dict) -> List[Dict]:
        # Your custom retrieval implementation
        pass
    
    def set_vectorstore(self, vectorstore):
        """Set the vector store to use."""
        self.vectorstore = vectorstore
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "retriever_type": "custom",
            "total_queries": self._query_count,
            "avg_latency": self._avg_latency
        }

# Register the retriever
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("retriever", "custom", CustomRetriever)
```

## Retrieval Pipeline

### Multi-stage Retrieval
Combine multiple retrievers in stages:

```json
{
  "retriever": {
    "type": "pipeline",
    "config": {
      "stages": [
        {
          "type": "bm25",
          "config": {"top_k": 50},
          "weight": 0.3
        },
        {
          "type": "similarity", 
          "config": {"top_k": 20},
          "weight": 0.7
        },
        {
          "type": "mmr",
          "config": {"top_k": 5, "lambda_diversity": 0.5},
          "weight": 1.0
        }
      ]
    }
  }
}
```

### Retrieval with Re-ranking
Add a re-ranking stage:

```python
retriever_config = {
    "type": "reranked",
    "config": {
        "base_retriever": "hybrid",
        "reranker": {
            "type": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 5
        },
        "initial_k": 20
    }
}
```

## Performance Optimization

### Caching
```json
{
  "retriever": {
    "config": {
      "cache_enabled": true,
      "cache_size": 1000,
      "cache_ttl": 3600,
      "cache_key_include_filters": true
    }
  }
}
```

### Parallel Retrieval
```json
{
  "retriever": {
    "config": {
      "parallel_search": true,
      "num_workers": 4,
      "batch_size": 10
    }
  }
}
```

### Query Optimization
```json
{
  "retriever": {
    "config": {
      "query_preprocessing": {
        "expand_synonyms": true,
        "correct_spelling": true,
        "remove_stopwords": false,
        "stem_terms": true
      }
    }
  }
}
```

## Evaluation and Metrics

### Retrieval Metrics
```python
from rag_engine.core.evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator()
metrics = evaluator.evaluate_retriever(
    retriever=retriever,
    test_queries=test_queries,
    ground_truth=ground_truth
)

print(f"Precision@5: {metrics['precision_at_5']:.3f}")
print(f"Recall@5: {metrics['recall_at_5']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"NDCG@5: {metrics['ndcg_at_5']:.3f}")
```

### A/B Testing
```python
retriever_config = {
    "ab_testing": {
        "enabled": true,
        "variants": {
            "A": {"type": "similarity", "weight": 0.5},
            "B": {"type": "hybrid", "weight": 0.5}
        },
        "metrics_to_track": ["latency", "relevance", "user_satisfaction"]
    }
}
```

## Filtering and Constraints

### Metadata Filtering
```python
# Filter by document type
results = retriever.retrieve(
    query="machine learning",
    filter={"document_type": "research_paper"}
)

# Complex filters
results = retriever.retrieve(
    query="python programming",
    filter={
        "$and": [
            {"language": "english"},
            {"$or": [
                {"difficulty": "beginner"},
                {"difficulty": "intermediate"}
            ]},
            {"date": {"$gte": "2024-01-01"}}
        ]
    }
)
```

### Time-based Constraints
```python
from datetime import datetime, timedelta

# Recent documents only
recent_filter = {
    "created_at": {
        "$gte": (datetime.now() - timedelta(days=30)).isoformat()
    }
}

results = retriever.retrieve(
    query="latest developments",
    filter=recent_filter
)
```

### Semantic Constraints
```python
# Ensure minimum relevance
results = retriever.retrieve(
    query="artificial intelligence",
    min_relevance_score=0.8,
    semantic_filters={
        "must_contain_concepts": ["AI", "machine learning"],
        "must_not_contain": ["science fiction", "movies"]
    }
)
```

## Monitoring and Analytics

### Query Analytics
```json
{
  "retriever": {
    "config": {
      "analytics": {
        "track_query_patterns": true,
        "log_zero_results": true,
        "measure_result_diversity": true,
        "track_user_interactions": true
      }
    }
  }
}
```

### Performance Monitoring
```python
# Monitor retrieval latency
retriever.monitor_latency(threshold_ms=500)

# Track result quality
retriever.track_result_quality(
    quality_threshold=0.7,
    alert_on_degradation=True
)

# Log slow queries
retriever.log_slow_queries(threshold_ms=1000)
```

## Troubleshooting

### Common Issues

**1. Poor Retrieval Quality**
```json
{
  "retriever": {
    "config": {
      "debug_mode": true,
      "explain_scores": true,
      "log_query_analysis": true
    }
  }
}
```

**2. Slow Retrieval**
```json
{
  "retriever": {
    "config": {
      "optimization": {
        "use_approximate_search": true,
        "early_termination": true,
        "result_caching": true
      }
    }
  }
}
```

**3. No Results Found**
```json
{
  "retriever": {
    "config": {
      "fallback_strategy": {
        "enabled": true,
        "lower_threshold": true,
        "expand_query": true,
        "use_fuzzy_matching": true
      }
    }
  }
}
```

**4. Inconsistent Results**
```json
{
  "retriever": {
    "config": {
      "consistency": {
        "deterministic_search": true,
        "stable_sorting": true,
        "cache_embeddings": true
      }
    }
  }
}
```

## Dependencies

Install required packages for different retrieval methods:

```bash
# BM25 support
pip install rank-bm25

# Cross-encoder re-ranking
pip install sentence-transformers

# Query expansion
pip install spacy nltk

# Advanced retrieval
pip install faiss-cpu scikit-learn
```
