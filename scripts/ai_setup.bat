@echo off
echo.
echo 🤖 RAG Engine AI-Powered Setup Assistant
echo ========================================
echo.
echo This will install a lightweight AI assistant that will help you
echo choose the perfect RAG Engine configuration for your needs!
echo.
echo The assistant will:
echo - Ask about your use case
echo - Recommend the best stack (DEMO/LOCAL/CLOUD/etc.)
echo - Install only what you need
echo - Provide ongoing support
echo.
pause

echo.
echo 📦 Setting up AI assistant...

REM Create scripts directory if it doesn't exist
if not exist "scripts" mkdir scripts

REM Install Python dependencies for the assistant
echo Installing assistant dependencies...
python -m pip install rich typer ollama-python

REM Run the AI setup assistant
echo.
echo 🚀 Starting AI assistant...
python scripts\ai_setup.py

echo.
echo ✅ Setup complete! The AI assistant is now available.
echo You can ask questions anytime with: rag-engine ask "your question"
echo.
pause
