# RAG Engine Quickstart

## Basic Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a configuration file** (`config.json`):
   ```json
   {
     "documents": [
       { "type": "txt", "path": "./examples/demo_document.md" }
     ],
     "chunking": {
       "method": "sentence", 
       "max_tokens": 300,
       "overlap": 30
     },
     "embedding": {
       "provider": "huggingface",
       "model": "sentence-transformers/all-MiniLM-L6-v2"
     },
     "vectorstore": {
       "provider": "chroma",
       "persist_directory": "./vectors"
     },
     "retrieval": {
       "method": "similarity",
       "top_k": 5
     },
     "prompting": {
       "type": "rag",
       "template_path": "./templates/rag_template.txt",
       "system_prompt": "You are a helpful AI assistant.",
       "context_window": 3000,
       "citation_format": "numbered"
     },
     "llm": {
       "provider": "openai",
       "model": "gpt-3.5-turbo",
       "temperature": 0.7
     }
   }
   ```

3. **Build the RAG pipeline:**
   ```bash
   python -m rag_engine build --config config.json
   ```

4. **Start chatting:**
   ```bash
   python -m rag_engine chat --config config.json
   ```

## Advanced Examples

### Conversational Assistant
Use the conversational prompter for multi-turn conversations:
```json
{
  "prompting": {
    "type": "conversational",
    "memory_length": 5,
    "context_compression": true,
    "persona": "helpful_teacher"
  }
}
```

### Code Assistant
For code explanation and debugging:
```json
{
  "prompting": {
    "type": "code_explanation",
    "language": "python",
    "include_syntax_highlighting": true,
    "add_comments": true
  }
}
```

### API Server
Start the enhanced API server:
```bash
python -m rag_engine serve --config config.json --orchestrator default
```

## Templates
The system includes pre-built templates in `templates/`:
- `rag_template.txt` - Standard RAG prompts
- `chat_template.txt` - Conversational prompts
- `chain_of_thought_template.txt` - Step-by-step reasoning

You can customize these or create your own templates!
