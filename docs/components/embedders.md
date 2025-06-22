# Text Embedders

Embedders convert text into dense vector representations that capture semantic meaning. These embeddings enable similarity search and are fundamental to RAG performance.

## Available Embedders

### HuggingFaceEmbedder
Uses Hugging Face transformers for text embedding.

**Configuration:**
```json
{
  "embedder": {
    "type": "huggingface",
    "config": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "device": "auto",
      "batch_size": 32,
      "max_length": 512,
      "normalize_embeddings": true
    }
  }
}
```

**Popular Models:**
- `all-MiniLM-L6-v2`: Fast, good general performance (384 dimensions)
- `all-mpnet-base-v2`: Better quality, slower (768 dimensions)
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A tasks
- `all-distilroberta-v1`: Balanced speed/quality

### OpenAIEmbedder
Uses OpenAI's embedding models via API.

**Configuration:**
```json
{
  "embedder": {
    "type": "openai",
    "config": {
      "model": "text-embedding-ada-002",
      "api_key": "${OPENAI_API_KEY}",
      "batch_size": 100,
      "max_retries": 3,
      "timeout": 30
    }
  }
}
```

**Models:**
- `text-embedding-ada-002`: High quality, 1536 dimensions
- `text-embedding-3-small`: Newer, efficient model
- `text-embedding-3-large`: Highest quality available

### LocalEmbedder
Uses locally hosted embedding models.

**Configuration:**
```json
{
  "embedder": {
    "type": "local",
    "config": {
      "model_path": "./models/sentence-transformer",
      "device": "cuda:0",
      "precision": "float16",
      "cache_embeddings": true
    }
  }
}
```

### CohereEmbedder
Uses Cohere's embedding API.

**Configuration:**
```json
{
  "embedder": {
    "type": "cohere",
    "config": {
      "model": "embed-english-v3.0",
      "api_key": "${COHERE_API_KEY}",
      "input_type": "search_document",
      "truncate": "END"
    }
  }
}
```

### GeminiVertexEmbedder
Uses Google's Gemini/Vertex AI embedding models. Supports both Gemini API and Vertex AI.

**Configuration (Gemini API):**
```json
{
  "embedder": {
    "type": "gemini",
    "config": {
      "model": "models/embedding-001",
      "api_key": "${GOOGLE_API_KEY}",
      "task_type": "retrieval_document",
      "batch_size": 100
    }
  }
}
```

**Configuration (Vertex AI):**
```json
{
  "embedder": {
    "type": "gemini",
    "config": {
      "use_vertex": true,
      "model": "textembedding-gecko@001",
      "project": "${GOOGLE_CLOUD_PROJECT}",
      "location": "us-central1",
      "batch_size": 100
    }
  }
}
```

**Configuration (Vertex AI with Service Account):**
```json
{
  "embedder": {
    "type": "gemini",
    "config": {
      "use_vertex": true,
      "model": "textembedding-gecko@001",
      "project": "your-gcp-project",
      "location": "us-central1",
      "credentials_path": "/path/to/service-account.json",
      "batch_size": 100
    }
  }
}
```

**Models:**
- `models/embedding-001`: Gemini embedding model
- `textembedding-gecko@001`: Vertex AI text embedding model
- `textembedding-gecko@003`: Latest Vertex AI model (higher quality)

### Authentication Options

**Google Cloud Authentication Methods:**

1. **Service Account File:**
   ```json
   {
     "credentials_path": "/path/to/service-account.json"
   }
   ```

2. **Environment Variable:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```

3. **Default Application Credentials:**
   ```bash
   gcloud auth application-default login
   ```

4. **Compute Engine/GKE Service Account:**
   - Automatically detected when running on GCP

## Usage Examples

### Basic Embedding
```python
from rag_engine.core.embedder import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder({
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
})

# Single text
text = "What is the capital of France?"
embedding = embedder.embed(text)
print(f"Embedding shape: {len(embedding)}")

# Multiple texts
texts = ["Hello world", "How are you?", "Fine, thank you"]
embeddings = embedder.embed_batch(texts)
print(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
```

### Document Processing
```python
from rag_engine.core.orchestration import ComponentRegistry

registry = ComponentRegistry()
embedder = registry.get_component("embedder", "openai", config)

documents = [
    {"content": "Document 1 content", "id": "doc1"},
    {"content": "Document 2 content", "id": "doc2"}
]

# Embed documents with metadata
embedded_docs = []
for doc in documents:
    embedding = embedder.embed(doc["content"])
    embedded_docs.append({
        **doc,
        "embedding": embedding,
        "embedding_model": embedder.model_name
    })
```

### Similarity Search
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compare embeddings
query_embedding = embedder.embed("What is machine learning?")
doc_embeddings = [embedder.embed(doc) for doc in documents]

# Calculate similarities
similarities = cosine_similarity(
    [query_embedding], 
    doc_embeddings
)[0]

# Get most similar documents
top_indices = np.argsort(similarities)[::-1][:5]
for idx in top_indices:
    print(f"Document {idx}: {similarities[idx]:.3f}")
```

## Model Selection Guide

### Performance Comparison

| Model | Dimensions | Speed | Quality | Memory |
|-------|------------|-------|---------|---------|
| `all-MiniLM-L6-v2` | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `all-mpnet-base-v2` | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `text-embedding-ada-002` | 1536 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | N/A (API) |
| `textembedding-gecko@001` | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A (API) |
| `multilingual-e5-large` | 1024 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

### Use Case Recommendations

**General Purpose RAG:**
```json
{
  "embedder": {
    "type": "huggingface",
    "config": {
      "model_name": "sentence-transformers/all-mpnet-base-v2"
    }
  }
}
```

**Fast Prototyping:**
```json
{
  "embedder": {
    "type": "huggingface", 
    "config": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

**Production Quality:**
```json
{
  "embedder": {
    "type": "openai",
    "config": {
      "model": "text-embedding-ada-002"
    }
  }
}
```

**Google Cloud/Vertex AI:**
```json
{
  "embedder": {
    "type": "gemini",
    "config": {
      "use_vertex": true,
      "model": "textembedding-gecko@001",
      "project": "your-gcp-project",
      "location": "us-central1"
    }
  }
}
```

**Multilingual Support:**
```json
{
  "embedder": {
    "type": "huggingface",
    "config": {
      "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    }
  }
}
```

## Creating Custom Embedders

Implement the `BaseEmbedder` interface:

```python
from rag_engine.core.base import BaseEmbedder
from typing import List, Union
import numpy as np

class CustomEmbedder(BaseEmbedder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model_name")
        self.dimension = config.get("dimension", 384)
        # Initialize your model
        
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        # Implement embedding logic
        embedding = self._encode_text(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        # Implement batch embedding
        embeddings = self._encode_texts(texts)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension
    
    def _encode_text(self, text: str) -> np.ndarray:
        # Your encoding implementation
        pass
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        # Your batch encoding implementation
        pass

# Register the embedder
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("embedder", "custom", CustomEmbedder)
```

## Advanced Configuration

### GPU Optimization
```json
{
  "embedder": {
    "config": {
      "device": "cuda:0",
      "precision": "float16",
      "enable_cpu_fallback": true,
      "memory_fraction": 0.8
    }
  }
}
```

### Batch Processing
```json
{
  "embedder": {
    "config": {
      "batch_size": 64,
      "adaptive_batching": true,
      "max_batch_size": 128,
      "batch_timeout": 1.0
    }
  }
}
```

### Caching
```json
{
  "embedder": {
    "config": {
      "cache_embeddings": true,
      "cache_directory": "./cache/embeddings",
      "cache_size_limit": "1GB",
      "cache_ttl": 86400
    }
  }
}
```

### Text Preprocessing
```json
{
  "embedder": {
    "config": {
      "preprocessing": {
        "lowercase": true,
        "remove_stopwords": false,
        "remove_punctuation": false,
        "max_length": 512,
        "truncation_strategy": "end"
      }
    }
  }
}
```

## Performance Optimization

### Memory Management
```python
embedder_config = {
    "memory_management": {
        "clear_cache_interval": 1000,
        "max_memory_usage": "4GB",
        "use_memory_mapping": True
    }
}
```

### Parallel Processing
```python
embedder_config = {
    "parallel_processing": {
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True
    }
}
```

### Model Quantization
```json
{
  "embedder": {
    "config": {
      "quantization": {
        "enabled": true,
        "method": "int8",
        "calibration_dataset": "small_sample"
      }
    }
  }
}
```

## Evaluation and Monitoring

### Quality Metrics
```python
from rag_engine.core.evaluation import EmbeddingEvaluator

evaluator = EmbeddingEvaluator()
metrics = evaluator.evaluate_embedder(
    embedder=embedder,
    test_queries=test_queries,
    ground_truth=ground_truth
)

print(f"Retrieval accuracy: {metrics['accuracy']:.3f}")
print(f"Average similarity: {metrics['avg_similarity']:.3f}")
```

### Performance Monitoring
```json
{
  "embedder": {
    "config": {
      "monitoring": {
        "track_latency": true,
        "track_memory": true,
        "log_slow_queries": true,
        "slow_query_threshold": 1.0
      }
    }
  }
}
```

## Multi-modal Embeddings

### Text + Image
```json
{
  "embedder": {
    "type": "multimodal",
    "config": {
      "text_model": "sentence-transformers/all-mpnet-base-v2",
      "image_model": "openai/clip-vit-base-patch32",
      "fusion_strategy": "concatenate"
    }
  }
}
```

### Specialized Domains
```json
{
  "embedder": {
    "type": "domain_specific",
    "config": {
      "domain": "biomedical",
      "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```json
{
  "embedder": {
    "config": {
      "batch_size": 16,
      "gradient_checkpointing": true,
      "cpu_offload": true
    }
  }
}
```

**2. Slow Inference**
```json
{
  "embedder": {
    "config": {
      "compile_model": true,
      "use_optimized_kernels": true,
      "enable_tensorrt": true
    }
  }
}
```

**3. Poor Quality Embeddings**
```json
{
  "embedder": {
    "config": {
      "fine_tuned_model": true,
      "domain_adaptation": true,
      "temperature_scaling": 0.1
    }
  }
}
```

**4. API Rate Limiting**
```json
{
  "embedder": {
    "config": {
      "rate_limit": 100,
      "retry_backoff": "exponential",
      "max_retries": 5,
      "fallback_embedder": "local"
    }
  }
}
```

## Dependencies

Install required packages based on embedder type:

```bash
# Hugging Face
pip install transformers torch sentence-transformers

# OpenAI
pip install openai

# Cohere
pip install cohere

# Google Gemini/Vertex AI
pip install google-generativeai
pip install google-cloud-aiplatform  # For Vertex AI support

# Optimization
pip install optimum onnxruntime
```
