# 🎯 Friend's House Demo - Complete Guide

**Show off your RAG Engine in 10 minutes!**

## Option A: Local LLM (Recommended - No API keys needed!)

### 1. Install Ollama (2 minutes)
```bash
# Download from: https://ollama.ai
# Or Windows: winget install Ollama.Ollama
```

### 2. Clone & Quick Setup (3 minutes)
```bash
git clone YOUR_REPO_URL
cd rag_engine

# Run the magic setup script
quick_setup.bat

# Or manual:
pip install -r requirements.txt
cd frontend && npm install && cd ..
cp examples/configs/demo_local_config.json config.json
```

### 3. Start Everything (1 minute)
```bash
# Terminal 1 - Backend
python -m rag_engine serve

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

### 4. Demo Time! (5 minutes)
1. Visit **http://localhost:3001** (or shown port)
2. **Toggle Dark/Light Mode** - Show off the theme system! 🌙✨
3. **Build Pipeline** - Go to Pipeline page, click "Build Pipeline"
4. **Upload Document** - Use the provided `demo_document.md`
5. **Chat** - Ask questions like:
   - "What is RAG?"
   - "How does this engine work?"
   - "What features does it have?"

---

## Option B: Cloud LLM (If local doesn't work)

### Quick Setup
```bash
cp examples/configs/demo_cloud_config.json config.json
# Edit config.json and add OpenAI API key
```

Same demo flow, but uses OpenAI instead of local model.

---

## Demo Script for Friends 🎬

**"Hey, check out this RAG system I built!"**

1. **"First, look at this modern UI"** - Show the dark theme, navigation
2. **"It's completely modular"** - Go to System page, show components
3. **"Watch this - I'll build a pipeline"** - Pipeline page, build it
4. **"Now I'll upload a document"** - Documents page, upload demo_document.md
5. **"And now I can chat with my document!"** - Chat page, ask questions
6. **"It even works with local models"** - Explain Ollama integration
7. **"Plus light/dark themes"** - Toggle the theme

**Total demo time: 3-5 minutes**

---

## Troubleshooting 🔧

**"Ollama not found"**
```bash
ollama serve  # Start Ollama server
ollama list   # Check installed models
```

**"Port already in use"**
- Frontend will auto-find available port
- Backend: Edit `config.json` → `api.port`

**"Model too slow"**
- Use `llama3.2:1b` (fastest small model)
- Or switch to cloud config with OpenAI

**"Dependencies missing"**
```bash
pip install -r requirements.txt
cd frontend && npm install
```

---

## What Makes This Cool? 💪

- **Dark Mode First** - Because engineers love dark themes
- **Modular Architecture** - Swap any component (LLM, embeddings, vector store)
- **Local & Cloud** - Works with Ollama or cloud APIs
- **Modern UI** - Vue 3 + Tailwind, responsive design
- **Real RAG** - Not just a chatbot, actual document retrieval
- **Experimental Badge** - Honest about being a WIP project

**Perfect conversation starter at any tech meetup!** 🚀
