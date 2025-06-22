# 🧪 Testing Your Instant Demo Setup

**Make sure everything works perfectly before showing your friend!**

## Test 0: Docker Desktop Check (Do This First!)

### Before any Docker tests:
```bash
# Check if Docker Desktop is running:
docker ps

# If you get "cannot connect" error:
# 1. Start Docker Desktop manually
# 2. Wait for it to fully start (green icon in system tray)
# 3. Then run tests
```

### Common Docker States:
- ✅ **Running**: `docker ps` shows containers
- ❌ **Not Running**: "cannot connect" error → Start Docker Desktop
- ❌ **Not Installed**: "docker is not recognized" → Install Docker Desktop

## Test 1: Docker Demo (Recommended Test)

### Quick Test (5 minutes)
```bash
# Clean start
docker-compose -f docker-compose.demo.yml down
docker-compose -f docker-compose.demo.yml up

# Wait for "ready" messages in the logs, then visit:
# http://localhost:3001
```

### What to Check:
- ✅ Frontend loads with dark theme
- ✅ Theme toggle works (moon/sun icon)
- ✅ Pipeline page shows components
- ✅ System page shows configuration
- ✅ Can build pipeline successfully

## Test 2: Local Development Test

### If you want to test the manual setup:
```bash
# Terminal 1 - Backend
python -m rag_engine serve

# Terminal 2 - Frontend  
cd frontend
npm run dev

# Visit: http://localhost:3001 (or shown port)
```

## Test 3: Windows Batch Script Test

### Test the one-click script:
```bash
# Save your current work, then test:
.\instant_demo.bat

# Should automatically:
# - Check Docker
# - Start services
# - Open browser
```

## Test 4: Fresh Clone Test (Most Important!)

### Simulate your friend's experience:
```bash
# Go to a different folder (not your dev folder)
cd C:\temp

# Clone as if you're at friend's house
git clone https://github.com/YOUR_USERNAME/rag_engine.git
cd rag_engine

# Test the instant demo
.\instant_demo.bat
```

## Test 5: Complete Demo Flow Test

### Test the actual demo you'll show:

1. **Theme Toggle** - Click sun/moon icon ✨
2. **Build Pipeline** - Go to Pipeline → "Build Pipeline" 
3. **Upload Document** - Documents → Upload `demo_document.md`
4. **Chat Test** - Chat → Ask: "What is RAG?"
5. **System View** - System → Show modular components

### Expected Results:
- Pipeline builds without errors
- Document uploads and shows chunks
- Chat responds with relevant info from document
- All pages look good in dark/light mode

## Test 6: Performance Test

### Check timing for friend demo:
```bash
# Time each step:
# 1. Git clone: ~30 seconds
# 2. Docker startup: ~3-4 minutes (first time)
# 3. Browser open: ~10 seconds
# 4. Demo flow: ~3-5 minutes

# Total: Under 10 minutes including setup
```

## Troubleshooting Common Issues

### Docker Issues:
```bash
# If containers don't start:
docker-compose -f docker-compose.demo.yml logs

# If Ollama model download fails:
docker-compose -f docker-compose.demo.yml exec ollama ollama pull llama3.2:1b

# Clean restart:
docker-compose -f docker-compose.demo.yml down --volumes
docker-compose -f docker-compose.demo.yml up --build
```

### Port Conflicts:
```bash
# Check what's using ports:
netstat -an | findstr ":3001"
netstat -an | findstr ":8000"
netstat -an | findstr ":11434"

# Kill processes if needed:
taskkill /F /PID <process_id>
```

### Frontend Not Loading:
```bash
# Check if backend is ready:
curl http://localhost:8000/health

# Check frontend logs:
docker-compose -f docker-compose.demo.yml logs frontend
```

## Pre-Friend Checklist ✅

Before going to your friend's house:

- [ ] Tested docker-compose.demo.yml works
- [ ] Tested instant_demo.bat works  
- [ ] Confirmed demo_document.md uploads properly
- [ ] Tested chat with sample questions
- [ ] Verified theme toggle works
- [ ] Checked all pages display correctly
- [ ] Timed the complete flow (should be < 10 min)
- [ ] Pushed latest changes to GitHub

## Success Criteria

Your demo is ready when:
- ✅ One command starts everything
- ✅ Browser opens automatically 
- ✅ Beautiful UI with working theme toggle
- ✅ Can build pipeline in < 1 minute
- ✅ Document upload and chat work
- ✅ Total setup time < 5 minutes

**If all tests pass, you're ready to wow your friend!** 🚀
