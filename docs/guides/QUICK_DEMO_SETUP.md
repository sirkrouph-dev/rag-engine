# 🚀 Quick Demo Setup - Local LLM

**Perfect for showing the RAG Engine at a friend's house!**

## 📦 Preset Stacks (Choose Your Adventure!)

We've designed preset configurations to minimize bloat and maximize ease of use:

| Stack | Abbreviation | Use Case | Dependencies | Setup Time |
|-------|--------------|----------|--------------|------------|
| **DEMO** | `demo` | Quick demos, friends' houses | Minimal (Ollama + basic) | ~5 min |
| **LOCAL** | `local` | Local development | Local LLMs + full features | ~10 min |
| **CLOUD** | `cloud` | Production with cloud APIs | OpenAI/Anthropic/etc. | ~3 min |
| **MINI** | `mini` | Embedded systems | Ultra-minimal | ~2 min |
| **FULL** | `full` | Research, all features | Everything included | ~15 min |
| **RESEARCH** | `research` | Academic/experimental | Cutting-edge models | ~20 min |

**Quick install examples:**
```bash
# Install specific stack
pip install -r requirements-demo.txt     # DEMO stack
pip install -r requirements-local.txt    # LOCAL stack  
pip install -r requirements-cloud.txt    # CLOUD stack

# Or use the configurator (coming soon)
rag-engine install demo                  # CLI command
```

**👉 For instant demos, use the DEMO stack below:**

---

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

## Option B: AI-Powered Setup (Ask the Engine!) 🤖✨

**Time: ~7 minutes | Let AI choose your perfect stack**

### The Crazy Idea: Your Personal RAG Consultant!

What if the RAG Engine itself helped you set it up? This option installs a lightweight local LLM that acts as your personal consultant, asking questions about your needs and recommending the perfect stack configuration.

### 1. Run the AI Setup Assistant

```bash
# Clone the repo
git clone https://github.com/yourusername/rag_engine.git
cd rag_engine

# Run the AI-powered setup (Windows)
.\ai_setup.bat

# Or manual (any OS)
python scripts/ai_setup.py
```

### 2. Chat with Your RAG Consultant

The script will install a tiny LLM (phi3.5 - only ~2GB) and start an interactive chat:

```
🤖 RAG Engine Assistant: Hi! I'm here to help you set up the perfect RAG configuration.

Let me ask you a few questions:

1. What's your main use case?
   a) Quick demo at a friend's house
   b) Local development and experimentation  
   c) Production deployment with cloud APIs
   d) Research with cutting-edge models
   e) Minimal embedded system

2. Do you prefer local LLMs or cloud APIs?

3. How much disk space can you spare?

4. What's your Python experience level?

Based on your answers, I'll recommend the perfect stack and install it for you!
```

### 3. Personalized Recommendations

The AI assistant knows about:
- **All stack configurations** (DEMO, LOCAL, CLOUD, MINI, FULL, RESEARCH)
- **Hardware requirements** for different models
- **Use case best practices**
- **Troubleshooting solutions**
- **Upgrade paths**

### 4. Ongoing Support

After setup, you can always ask:
```bash
# Ask the engine anything!
rag-engine ask "How do I add more documents?"
rag-engine ask "Which LLM model is best for my use case?"
rag-engine ask "How do I switch to cloud APIs?"
rag-engine ask "My setup is slow, how can I optimize it?"
rag-engine ask "What's the difference between LOCAL and CLOUD stacks?"
```

**Example conversation:**
```
You: "I want to demo this at my friend's house but their internet is slow"
🤖: "Perfect! I recommend the DEMO stack with phi3.5 model. It's only 2GB, 
     runs offline, and installs in 5 minutes. Would you like me to set 
     that up for you?"

You: rag-engine ask "How do I make the responses faster?"
🤖: "For faster responses, try:
     1. Use llama3.2:1b (smallest model)
     2. Reduce chunk_size in config 
     3. Use simple embeddings instead of all-MiniLM
     4. Enable GPU acceleration if available"
```

---

## Option D: Windows One-Click (Ultra Fast!) ⚡

**Time: ~5 minutes | Perfect for demos**

1. **Download and run the instant demo:**
   ```cmd
   .\instant_demo.bat
   ```

2. **What this does:**
   - Installs DEMO stack dependencies automatically
   - Downloads Ollama if needed
   - Pulls a lightweight model (phi3.5 or llama3.2)
   - Starts the RAG engine with demo config
   - Opens your browser automatically

3. **Demo features:**
   - Document upload and Q&A
   - Real-time chat with local LLM
   - No cloud dependencies
   - Works offline

See `INSTANT_DEMO.md` for detailed instructions.

---

## Option E: Manual Setup (Python - DEMO Stack) 🐍

**Time: ~8 minutes | Uses DEMO preset stack**

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

# Install DEMO stack dependencies (minimal bloat!)
pip install -r requirements-demo.txt

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

**🤖 Ask the AI Assistant first:**
```bash
rag-engine ask "I'm having trouble with setup, what should I check?"
rag-engine ask "Ollama is not working, how do I fix it?"
rag-engine ask "The frontend won't start, what's wrong?"
```

**Common issues:**

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
- Ask the assistant: `rag-engine ask "How can I make my model faster?"`

**Setup script failed?**
- Ask the assistant: `rag-engine ask "My setup failed, what went wrong?"`
- Check the logs in the terminal output

---

## 🔄 Want to Switch Stacks Later?

After the demo, you can easily upgrade or change stacks:

```bash
# Upgrade to LOCAL stack (more features)
pip install -r requirements-local.txt

# Or switch to CLOUD stack (production APIs)
pip install -r requirements-cloud.txt

# Or install everything (FULL stack)
pip install -r requirements-full.txt
```

**Coming soon:** CLI commands for easier stack management:
```bash
rag-engine switch-to local    # Switch to LOCAL stack
rag-engine add cloud-apis     # Add cloud API support
rag-engine install research   # Install RESEARCH stack
```

## 📊 Stack Comparison

| Feature | DEMO | LOCAL | CLOUD | MINI | FULL |
|---------|------|-------|-------|------|------|
| Local LLMs | ✅ | ✅ | ❌ | ❌ | ✅ |
| Cloud APIs | ❌ | ✅ | ✅ | ❌ | ✅ |
| Vector DBs | Simple | Multiple | Multiple | None | All |
| Web UI | ✅ | ✅ | ✅ | ❌ | ✅ |
| Plugins | Basic | ✅ | ✅ | ❌ | ✅ |
| Install Size | ~200MB | ~500MB | ~100MB | ~50MB | ~1GB |

---

**Total setup time: ~10 minutes | Demo time: ~5 minutes** 🎯
