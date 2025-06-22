# Examples

This directory contains example configurations and scripts for the RAG Engine.

> **⚠️ EXPERIMENTAL EXAMPLES ⚠️**
> 
> **These examples are for development and experimentation only. Not recommended for production use.**

## 📁 Directory Structure

```
examples/
├── configs/          # Example configuration files
│   ├── basic.json           # Basic RAG setup
│   ├── production.json      # Advanced configuration example
│   ├── vertex_ai.json       # Google Vertex AI configuration
│   └── hybrid.json          # Hybrid retrieval setup
├── scripts/          # Example Python scripts
│   └── vertex_ai_example.py # Vertex AI setup and usage
└── quickstart.md     # Quick start guide
```

## 🚀 Quick Start

### 1. Basic Configuration
```bash
# Use the basic example configuration
python -m rag_engine serve --config examples/configs/example_config.json
```

### 2. Vertex AI Example
```bash
# Set up Vertex AI embeddings
python examples/scripts/vertex_ai_example.py

# Use Vertex AI configuration
python -m rag_engine serve --config examples/configs/vertex_ai_example.json
```

### 3. Hybrid Retrieval
```bash
# Use hybrid retrieval configuration
python -m rag_engine serve --config examples/configs/enhanced_production.json
```

## 📋 Configuration Files

### Basic Setup (`example_config.json`)
- Simple document loading
- HuggingFace embeddings
- ChromaDB vector store
- OpenAI LLM

### Advanced Setup (`enhanced_production.json`)
- Multiple document sources
- Hybrid retrieval (semantic + BM25)
- Production-like configuration
- Advanced features enabled

### Vertex AI Setup (`vertex_ai_example.json`)
- Google Cloud Vertex AI embeddings
- Service account authentication
- Optimized for GCP environments

## 🔧 Customization

All example configurations can be customized by:

1. **Copying** an example file
2. **Modifying** the parameters for your use case
3. **Running** with your custom config:
   ```bash
   python -m rag_engine serve --config your_custom_config.json
   ```

## 📚 Documentation

For detailed configuration options, see:
- [Configuration Guide](../docs/configuration.md)
- [Component Documentation](../docs/components/)
- [Deployment Guide](../docs/deployment/)

## ⚠️ Important Notes

- These examples are for **development and testing** only
- **Do not use in production** without proper security review
- Update API keys and credentials before use
- Test thoroughly in your environment before deployment
