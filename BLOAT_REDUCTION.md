# 📦 RAG Engine - Dependency Bloat Reduction Strategy

## Current Problem Analysis

Our `requirements.txt` has **major bloat issues**:
- 🔥 **PyTorch** (~2GB): Only needed for local transformers
- 🔥 **Multiple web frameworks**: FastAPI, Flask, Django, Streamlit, Gradio
- 🔥 **All vector stores**: ChromaDB, FAISS, Pinecone, Qdrant
- 🔥 **All LLM providers**: OpenAI, Google, Transformers, Ollama
- 🔥 **Heavy NLP**: NLTK, sentence-transformers

**Total install size**: Likely 3-5GB+ and 10+ minutes

## Solution: Modular Installation Strategy

### 1. Preset Stack Abbreviations 🎯

#### **DEMO** - *Instant Demo Stack*
```yaml
# Perfect for: Showing friends, quick prototypes
# Size: ~50MB | Setup: 2 minutes
components:
  llm: ollama (llama3.2:1b)
  embeddings: sentence-transformers (mini)
  vectordb: simple (in-memory)
  ui: minimal frontend
install: "pip install rag-engine[demo]"
```

#### **LOCAL** - *Local Development Stack*
```yaml
# Perfect for: Privacy-focused, offline development  
# Size: ~300MB | Setup: 5 minutes
components:
  llm: ollama (phi3, llama)
  embeddings: sentence-transformers (full)
  vectordb: faiss
  ui: full frontend + system view
install: "pip install rag-engine[local]"
```

#### **CLOUD** - *Cloud-Powered Stack*
```yaml
# Perfect for: Production, best performance
# Size: ~80MB | Setup: 1 minute (just API keys)
components:
  llm: openai (gpt-4o, claude)
  embeddings: openai (ada-002)
  vectordb: pinecone/chromadb
  ui: full frontend + analytics
install: "pip install rag-engine[cloud]"
```

#### **MINI** - *Minimal Core Stack*
```yaml
# Perfect for: Embedding in other apps, custom builds
# Size: ~15MB | Setup: 30 seconds
components:
  llm: none (API only)
  embeddings: basic
  vectordb: simple
  ui: API only
install: "pip install rag-engine[mini]"
```

#### **FULL** - *Everything Stack*
```yaml
# Perfect for: Developers, researchers, experimentation
# Size: ~1.5GB | Setup: 10 minutes
components:
  llm: all (openai, ollama, transformers)
  embeddings: all providers
  vectordb: all stores
  ui: all interfaces
install: "pip install rag-engine[full]"
```

#### **RESEARCH** - *Academic/Research Stack*
```yaml
# Perfect for: Papers, experiments, benchmarking
# Size: ~800MB | Setup: 8 minutes
components:
  llm: transformers (local models)
  embeddings: multiple (comparison)
  vectordb: faiss + chromadb
  ui: jupyter + frontend + metrics
install: "pip install rag-engine[research]"
```

### 2. Smart Installation Commands
```bash
# One-command presets
rag-engine install demo
rag-engine install local --gpu
rag-engine install cloud --provider openai

# Mix and match
pip install rag-engine[local,cloud]  # Best of both
pip install rag-engine[mini,research]  # Custom combo

# Easy switching
rag-engine switch-to cloud
rag-engine add research --extend-current
```

### 3. Core Minimal Requirements
**Goal**: Get basic RAG working in <1 minute, <100MB

### 2. Optional Extra Packages
**Strategy**: Install only what you actually use

### 3. Pre-built Docker Images
**Strategy**: Different images for different use cases

### 4. Lightweight Alternatives
**Strategy**: Replace heavy deps with lighter ones

## Implementation Plan

1. **Split requirements.txt** into multiple files
2. **Create setup.py with extras** for optional components  
3. **Build lightweight Docker images**
4. **Add dependency check system** that warns about missing packages
5. **Create installation wizard** that asks what you want

This will transform the experience from:
- ❌ `pip install -r requirements.txt` (3GB, 10 minutes, often fails)
- ✅ `pip install rag-engine` (50MB, 30 seconds, works everywhere)

Then optionally add what you need:
- `pip install rag-engine[local-llm]` for Ollama/Transformers
- `pip install rag-engine[cloud]` for OpenAI/Google
- `pip install rag-engine[vector-stores]` for ChromaDB/FAISS

**Result**: Friends can try it instantly, developers can add features as needed!
