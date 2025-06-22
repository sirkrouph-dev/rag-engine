# Large Language Models (LLMs)

LLMs are the core reasoning engines in RAG systems, responsible for generating responses based on retrieved context and user queries. The RAG Engine supports multiple LLM providers and local models.

## Available LLMs

### OpenAI Models
Access to GPT models via OpenAI API.

**Configuration:**
```json
{
  "llm": {
    "type": "openai",
    "config": {
      "model": "gpt-3.5-turbo",
      "api_key": "${OPENAI_API_KEY}",
      "temperature": 0.7,
      "max_tokens": 1500,
      "top_p": 1.0,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
  }
}
```

**Available Models:**
- `gpt-3.5-turbo`: Fast, cost-effective
- `gpt-4`: Higher quality, more expensive
- `gpt-4-turbo`: Latest GPT-4 variant
- `gpt-4o`: Optimized for speed

### Anthropic Claude
Claude models via Anthropic API.

**Configuration:**
```json
{
  "llm": {
    "type": "anthropic",
    "config": {
      "model": "claude-3-haiku-20240307",
      "api_key": "${ANTHROPIC_API_KEY}",
      "max_tokens": 1500,
      "temperature": 0.7,
      "top_p": 1.0
    }
  }
}
```

**Available Models:**
- `claude-3-haiku-20240307`: Fast and efficient
- `claude-3-sonnet-20240229`: Balanced performance
- `claude-3-opus-20240229`: Highest capability

### Local LLMs via Ollama
Run models locally using Ollama.

**Configuration:**
```json
{
  "llm": {
    "type": "ollama",
    "config": {
      "model": "llama2:7b",
      "base_url": "http://localhost:11434",
      "temperature": 0.7,
      "num_predict": 1500,
      "top_k": 40,
      "top_p": 0.9
    }
  }
}
```

**Popular Models:**
- `llama2:7b`: Good general performance
- `mistral:7b`: Excellent instruction following
- `codellama:7b`: Specialized for code
- `vicuna:13b`: High-quality chat model

### Hugging Face Transformers
Local models via Hugging Face.

**Configuration:**
```json
{
  "llm": {
    "type": "huggingface",
    "config": {
      "model_name": "microsoft/DialoGPT-medium",
      "device": "auto",
      "torch_dtype": "float16",
      "max_new_tokens": 1500,
      "temperature": 0.7,
      "do_sample": true
    }
  }
}
```

### Azure OpenAI
OpenAI models via Azure.

**Configuration:**
```json
{
  "llm": {
    "type": "azure_openai",
    "config": {
      "deployment_name": "gpt-35-turbo",
      "api_key": "${AZURE_OPENAI_API_KEY}",
      "api_base": "${AZURE_OPENAI_ENDPOINT}",
      "api_version": "2024-02-15-preview",
      "temperature": 0.7,
      "max_tokens": 1500
    }
  }
}
```

## Usage Examples

### Basic Text Generation
```python
from rag_engine.core.llm import OpenAILLM

llm = OpenAILLM({
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
})

# Generate response
response = llm.generate(
    prompt="Explain quantum computing in simple terms",
    max_tokens=200
)
print(response)
```

### RAG Response Generation
```python
# Generate response with context
context_documents = [
    {"content": "Quantum computing uses quantum bits...", "score": 0.9},
    {"content": "Classical computers use binary bits...", "score": 0.8}
]

prompt = llm.create_rag_prompt(
    query="What is quantum computing?",
    context_documents=context_documents
)

response = llm.generate(prompt)
```

### Chat Conversation
```python
# Maintain conversation history
conversation = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give me examples?"}
]

response = llm.chat(conversation)
```

### Streaming Responses
```python
# Stream response tokens
for token in llm.stream_generate("Tell me about AI"):
    print(token, end="", flush=True)
```

## Prompt Engineering

### RAG Prompt Templates
```python
rag_prompt_template = """
You are a helpful AI assistant. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Cite relevant parts of the context when appropriate

Answer:
"""
```

### Custom Prompt Templates
```json
{
  "llm": {
    "config": {
      "prompt_templates": {
        "rag": "custom_rag_template.txt",
        "chat": "custom_chat_template.txt",
        "summarization": "custom_summary_template.txt"
      }
    }
  }
}
```

### Dynamic Prompt Construction
```python
def create_dynamic_prompt(query, documents, user_context):
    context = "\n".join([doc["content"] for doc in documents])
    
    if user_context.get("expertise_level") == "beginner":
        instruction = "Explain in simple terms suitable for beginners."
    else:
        instruction = "Provide a detailed technical explanation."
    
    return f"""
    Context: {context}
    
    Question: {query}
    
    {instruction}
    
    Answer:
    """
```

## Creating Custom LLMs

Implement the `BaseLLM` interface:

```python
from rag_engine.core.base import BaseLLM
from typing import Dict, Any, List, Iterator, Optional

class CustomLLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1500)
        # Initialize your model
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        
        # Implement generation logic
        response = self._generate_text(prompt, max_tokens, temperature)
        return response
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response for chat conversation."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, **kwargs)
    
    def stream_generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> Iterator[str]:
        """Stream generation tokens."""
        # Implement streaming logic
        for token in self._stream_tokens(prompt, **kwargs):
            yield token
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Implement token counting
        return len(text.split())  # Simple approximation
    
    def _generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Your model-specific generation logic
        pass
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Convert chat messages to prompt format
        pass

# Register the LLM
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("llm", "custom", CustomLLM)
```

## Advanced Configuration

### Model Parameters
```json
{
  "llm": {
    "config": {
      "generation_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "num_beams": 1,
        "early_stopping": false
      }
    }
  }
}
```

### Context Management
```json
{
  "llm": {
    "config": {
      "context_management": {
        "max_context_length": 4096,
        "context_overflow_strategy": "truncate_middle",
        "preserve_system_message": true,
        "conversation_memory": 10
      }
    }
  }
}
```

### Safety and Filtering
```json
{
  "llm": {
    "config": {
      "safety": {
        "content_filter": true,
        "toxicity_threshold": 0.8,
        "prompt_injection_detection": true,
        "output_sanitization": true
      }
    }
  }
}
```

## Performance Optimization

### Batching
```json
{
  "llm": {
    "config": {
      "batching": {
        "enabled": true,
        "max_batch_size": 10,
        "batch_timeout": 0.1,
        "dynamic_batching": true
      }
    }
  }
}
```

### Caching
```json
{
  "llm": {
    "config": {
      "caching": {
        "enabled": true,
        "cache_size": 1000,
        "cache_ttl": 3600,
        "semantic_caching": true,
        "similarity_threshold": 0.95
      }
    }
  }
}
```

### GPU Optimization
```json
{
  "llm": {
    "config": {
      "gpu_optimization": {
        "device": "cuda:0",
        "precision": "fp16",
        "gradient_checkpointing": true,
        "model_parallelism": "auto"
      }
    }
  }
}
```

## Model Comparison

| Provider | Model | Context Length | Speed | Quality | Cost |
|----------|-------|----------------|-------|---------|------|
| **OpenAI** | GPT-3.5 Turbo | 16K | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **OpenAI** | GPT-4 | 128K | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Anthropic** | Claude-3 Haiku | 200K | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Anthropic** | Claude-3 Opus | 200K | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Local** | Llama2 7B | 4K | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Local** | Mistral 7B | 8K | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Error Handling and Fallbacks

### Retry Logic
```json
{
  "llm": {
    "config": {
      "retry": {
        "max_retries": 3,
        "retry_delay": 1.0,
        "exponential_backoff": true,
        "retry_on_timeout": true,
        "retry_on_rate_limit": true
      }
    }
  }
}
```

### Fallback Models
```json
{
  "llm": {
    "config": {
      "fallback": {
        "enabled": true,
        "fallback_models": [
          {"type": "openai", "model": "gpt-3.5-turbo"},
          {"type": "ollama", "model": "llama2:7b"}
        ],
        "fallback_triggers": ["timeout", "rate_limit", "error"]
      }
    }
  }
}
```

### Error Recovery
```python
try:
    response = llm.generate(prompt)
except RateLimitError:
    # Wait and retry
    time.sleep(60)
    response = llm.generate(prompt)
except ModelOverloadedError:
    # Switch to fallback model
    fallback_llm = get_fallback_llm()
    response = fallback_llm.generate(prompt)
```

## Monitoring and Analytics

### Usage Tracking
```json
{
  "llm": {
    "config": {
      "monitoring": {
        "track_token_usage": true,
        "track_latency": true,
        "track_cost": true,
        "log_requests": true
      }
    }
  }
}
```

### Quality Metrics
```python
from rag_engine.core.evaluation import LLMEvaluator

evaluator = LLMEvaluator()
metrics = evaluator.evaluate_llm(
    llm=llm,
    test_cases=test_cases,
    metrics=["coherence", "relevance", "factuality"]
)

print(f"Coherence: {metrics['coherence']:.3f}")
print(f"Relevance: {metrics['relevance']:.3f}")
print(f"Factuality: {metrics['factuality']:.3f}")
```

### Cost Tracking
```python
# Track API costs
cost_tracker = llm.get_cost_tracker()
print(f"Total cost: ${cost_tracker.total_cost:.2f}")
print(f"Tokens used: {cost_tracker.total_tokens:,}")
print(f"Requests: {cost_tracker.total_requests:,}")
```

## Security Considerations

### API Key Management
```json
{
  "llm": {
    "config": {
      "security": {
        "api_key_rotation": true,
        "encrypt_keys": true,
        "use_environment_vars": true,
        "audit_api_calls": true
      }
    }
  }
}
```

### Content Filtering
```python
llm_config = {
    "content_filter": {
        "enabled": true,
        "filter_input": true,
        "filter_output": true,
        "toxicity_threshold": 0.8,
        "pii_detection": true
    }
}
```

### Prompt Injection Prevention
```json
{
  "llm": {
    "config": {
      "prompt_security": {
        "injection_detection": true,
        "sanitize_input": true,
        "instruction_isolation": true,
        "output_validation": true
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Rate Limiting**
```json
{
  "llm": {
    "config": {
      "rate_limiting": {
        "requests_per_minute": 60,
        "tokens_per_minute": 40000,
        "concurrent_requests": 5
      }
    }
  }
}
```

**2. Context Length Exceeded**
```json
{
  "llm": {
    "config": {
      "context_management": {
        "auto_truncate": true,
        "truncation_strategy": "middle",
        "preserve_query": true
      }
    }
  }
}
```

**3. Poor Response Quality**
```json
{
  "llm": {
    "config": {
      "quality_improvement": {
        "use_few_shot_examples": true,
        "enable_chain_of_thought": true,
        "temperature_adjustment": "auto"
      }
    }
  }
}
```

**4. High Latency**
```json
{
  "llm": {
    "config": {
      "performance": {
        "enable_streaming": true,
        "use_smaller_model": "auto",
        "optimize_prompts": true
      }
    }
  }
}
```

## Dependencies

Install required packages for different LLM providers:

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Hugging Face
pip install transformers torch

# Ollama
pip install ollama

# Azure OpenAI
pip install openai azure-identity
```
