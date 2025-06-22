# Prompt Templates and Processors

Prompters handle the construction and processing of prompts for LLMs. They manage templates, context formatting, and prompt optimization to ensure effective communication with language models.

## Available Prompters

### BasicPrompter
Simple template-based prompt construction.

**Configuration:**
```json
{
  "prompter": {
    "type": "basic",
    "config": {
      "template": "rag_template",
      "max_context_length": 4000,
      "context_separator": "\n---\n",
      "include_metadata": true
    }
  }
}
```

### RAGPrompter
Specialized for RAG applications with context management.

**Configuration:**
```json
{
  "prompter": {
    "type": "rag",
    "config": {
      "template_path": "./templates/rag_template.txt",
      "context_window": 3000,
      "query_enhancement": true,
      "citation_format": "numbered",
      "fallback_template": "simple"
    }
  }
}
```

### ConversationalPrompter
Manages multi-turn conversations with memory.

**Configuration:**
```json
{
  "prompter": {
    "type": "conversational",
    "config": {
      "memory_length": 10,
      "context_compression": true,
      "persona": "helpful_assistant",
      "conversation_template": "chat_template.txt"
    }
  }
}
```

### ChainOfThoughtPrompter
Implements chain-of-thought reasoning patterns.

**Configuration:**
```json
{
  "prompter": {
    "type": "chain_of_thought",
    "config": {
      "reasoning_steps": 3,
      "explicit_reasoning": true,
      "step_separator": "\nStep {n}:",
      "conclusion_prompt": "Therefore:"
    }
  }
}
```

## Default Templates

### RAG Template
```text
You are a helpful AI assistant. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
- Base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Be precise and cite relevant parts when appropriate
- Do not make up information not present in the context

Answer:
```

### Chat Template
```text
You are a knowledgeable and helpful AI assistant. You have access to the following information to help answer questions:

{context}

Current conversation:
{conversation_history}

User: {query}

Assistant: {last_response}

User: {query}
```

### Chain of Thought Template
```text
Let's think about this step by step.

Question: {query}

Context:
{context}

Step 1: Understanding the question
{step_1_analysis}

Step 2: Analyzing the context
{step_2_analysis}

Step 3: Reasoning through the answer
{step_3_reasoning}

Therefore: {conclusion}
```

## Usage Examples

### Basic Prompt Construction
```python
from rag_engine.core.prompting import RAGPrompter

prompter = RAGPrompter({
    "template_path": "./templates/rag_template.txt",
    "context_window": 3000
})

# Format a prompt
context_documents = [
    {"content": "Python is a programming language...", "score": 0.9},
    {"content": "Machine learning uses algorithms...", "score": 0.8}
]

prompt = prompter.format_prompt(
    query="What is Python used for?",
    context_documents=context_documents
)

print(prompt)
```

### Conversational Prompting
```python
from rag_engine.core.prompting import ConversationalPrompter

prompter = ConversationalPrompter({
    "memory_length": 5,
    "context_compression": True
})

# Add conversation history
prompter.add_message("user", "What is machine learning?")
prompter.add_message("assistant", "Machine learning is a subset of AI...")
prompter.add_message("user", "Can you give me examples?")

# Generate prompt with context
prompt = prompter.format_prompt(
    query="Can you give me examples?",
    context_documents=context_docs,
    include_history=True
)
```

### Custom Template Usage
```python
custom_template = """
You are an expert in {domain}. 

Context: {context}

Question: {query}

Provide a detailed answer based on your expertise and the given context.
"""

prompter = BasicPrompter({
    "custom_template": custom_template,
    "template_variables": {
        "domain": "machine learning"
    }
})
```

## Creating Custom Prompters

Implement the `BasePrompter` interface:

```python
from rag_engine.core.base import BasePrompter
from typing import Dict, Any, List, Optional

class CustomPrompter(BasePrompter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.template = config.get("template", "")
        self.max_context_length = config.get("max_context_length", 4000)
        
    def format_prompt(
        self,
        query: str,
        context_documents: List[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Format the prompt with query and context."""
        
        # Process context documents
        context = self._format_context(context_documents or [])
        
        # Apply template
        prompt = self.template.format(
            query=query,
            context=context,
            **kwargs
        )
        
        # Ensure length constraints
        return self._truncate_if_needed(prompt)
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format context documents into a string."""
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            
            # Add citation number and content
            context_part = f"[{i+1}] {content}"
            if self.config.get("include_scores", False):
                context_part += f" (Score: {score:.3f})"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _truncate_if_needed(self, prompt: str) -> str:
        """Truncate prompt if it exceeds max length."""
        if len(prompt) <= self.max_context_length:
            return prompt
        
        # Smart truncation logic here
        return prompt[:self.max_context_length]

# Register the prompter
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("prompter", "custom", CustomPrompter)
```

## Advanced Features

### Dynamic Template Selection
```json
{
  "prompter": {
    "type": "adaptive",
    "config": {
      "template_selection": {
        "strategy": "query_type",
        "templates": {
          "factual": "factual_template.txt",
          "creative": "creative_template.txt",
          "analytical": "analytical_template.txt"
        }
      }
    }
  }
}
```

### Context Optimization
```json
{
  "prompter": {
    "config": {
      "context_optimization": {
        "relevance_filtering": true,
        "diversity_enhancement": true,
        "redundancy_removal": true,
        "length_balancing": true
      }
    }
  }
}
```

### Multi-language Support
```json
{
  "prompter": {
    "config": {
      "multilingual": {
        "auto_detect_language": true,
        "templates_by_language": {
          "en": "templates/english/",
          "es": "templates/spanish/",
          "fr": "templates/french/"
        }
      }
    }
  }
}
```

## Template Management

### Template Directory Structure
```
templates/
├── rag/
│   ├── basic.txt
│   ├── detailed.txt
│   └── concise.txt
├── chat/
│   ├── casual.txt
│   ├── professional.txt
│   └── technical.txt
└── specialized/
    ├── medical.txt
    ├── legal.txt
    └── scientific.txt
```

### Template Variables
```json
{
  "prompter": {
    "config": {
      "global_variables": {
        "assistant_name": "RAG Assistant",
        "system_date": "{current_date}",
        "domain": "general"
      },
      "variable_processors": {
        "current_date": "auto_generate",
        "user_context": "extract_from_session"
      }
    }
  }
}
```

### Template Validation
```python
def validate_template(template_content: str) -> Dict[str, Any]:
    """Validate template format and variables."""
    required_vars = ["query", "context"]
    optional_vars = ["conversation_history", "metadata"]
    
    validation_result = {
        "valid": True,
        "missing_required": [],
        "unused_optional": [],
        "custom_vars": []
    }
    
    # Check for required variables
    for var in required_vars:
        if f"{{{var}}}" not in template_content:
            validation_result["missing_required"].append(var)
            validation_result["valid"] = False
    
    return validation_result
```

## Performance Optimization

### Template Caching
```json
{
  "prompter": {
    "config": {
      "caching": {
        "enabled": true,
        "cache_compiled_templates": true,
        "cache_formatted_prompts": false,
        "cache_size": 100
      }
    }
  }
}
```

### Batch Processing
```python
# Process multiple queries efficiently
prompter = RAGPrompter(config)

queries = ["What is AI?", "How does ML work?", "Explain neural networks"]
context_docs_list = [docs1, docs2, docs3]

prompts = prompter.format_prompts_batch(
    queries=queries,
    context_documents_list=context_docs_list
)
```

### Memory Management
```json
{
  "prompter": {
    "config": {
      "memory_management": {
        "max_conversation_length": 10,
        "compression_strategy": "summarization",
        "cleanup_interval": 100
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. Template Not Found**
```json
{
  "prompter": {
    "config": {
      "fallback_template": "default_rag.txt",
      "template_search_paths": [
        "./templates/",
        "./custom_templates/",
        "/system/templates/"
      ]
    }
  }
}
```

**2. Context Too Long**
```json
{
  "prompter": {
    "config": {
      "context_management": {
        "auto_truncate": true,
        "truncation_strategy": "smart",
        "preserve_query": true,
        "preserve_citations": true
      }
    }
  }
}
```

**3. Poor Prompt Quality**
```json
{
  "prompter": {
    "config": {
      "quality_enhancement": {
        "add_examples": true,
        "clarify_instructions": true,
        "enhance_context": true
      }
    }
  }
}
```
