# ⚡ INSTANT RAG Demo - No Setup Required!

**Zero-to-demo in 5 minutes - perfect for friends without Python!**

## Option A: Docker (Easiest - No Python needed!) ⭐

### 1. Prerequisites (2 minutes)
```bash
# Install Docker Desktop: https://docker.com/products/docker-desktop
# On Windows: winget install Docker.DockerDesktop
```

### 2. One-Command Setup (3 minutes)
```bash
git clone https://github.com/yourusername/rag_engine.git
cd rag_engine

# This starts everything (backend + frontend + Ollama)
docker-compose up
```

**That's it!** Visit: **http://localhost:3001**

The Docker setup includes:
- ✅ Python environment (pre-configured)
- ✅ All dependencies installed
- ✅ Ollama with llama3.2:1b model
- ✅ Frontend and backend ready
- ✅ No manual configuration needed

---

## Option B: Manual Setup (If Python already installed)

### 1. Prerequisites (5 minutes)

#### Install Ollama (easiest local LLM)
```bash
# Download and install from: https://ollama.ai
# Or use winget on Windows:
winget install Ollama.Ollama
```

#### Pull a small, fast model
```bash
# Llama 3.2 1B - very small and fast (1.3GB)
ollama pull llama3.2:1b

# Alternative: Phi-3 Mini (2.3GB)  
ollama pull phi3:mini
```

### 2. Clone and Setup (3 minutes)

```bash
git clone https://github.com/yourusername/rag_engine.git
cd rag_engine

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies  
cd frontend
npm install
cd ..
```

### 3. Quick Configuration (1 minute)

Copy the demo config:
```bash
cp examples/configs/demo_local_config.json config.json
```

This uses:
- **Ollama** for LLM (llama3.2:1b)
- **Simple embeddings** (sentence-transformers)
- **In-memory vector store** (no setup needed)
- **Basic chunking** (fast)

### 4. Run Demo (30 seconds)

#### Terminal 1 - Backend:
```bash
python -m rag_engine serve
```

#### Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Visit: **http://localhost:3001** (or shown port)

### 5. Quick Demo Flow (2 minutes)

1. **Dashboard** - Show the dark/light theme toggle ✨
2. **Pipeline** - Click "Build Pipeline" 
3. **Documents** - Upload a small text file
4. **Chat** - Ask questions about the document
5. **System** - Show the configuration

## Troubleshooting

**Ollama not found?**
```bash
# Make sure Ollama is running
ollama serve
```

**Port conflicts?**
- Backend: Change port in config.json
- Frontend: Will auto-find available port

**Model too slow?**
- Use `llama3.2:1b` (fastest)
- Or switch to OpenAI API in config

---

**Total setup time: ~5 minutes | Demo time: ~3 minutes** ⚡
