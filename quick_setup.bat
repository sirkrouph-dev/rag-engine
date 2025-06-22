@echo off
echo 🚀 RAG Engine Quick Demo Setup
echo ================================

echo.
echo 1. Checking if Ollama is installed...
where ollama >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama found!
    echo.
    echo 2. Pulling small model (llama3.2:1b)...
    ollama pull llama3.2:1b
    echo.
    echo 3. Using local Ollama config...
    copy examples\configs\demo_local_config.json config.json
) else (
    echo ❌ Ollama not found. Using OpenAI config...
    echo Please edit config.json and add your OpenAI API key.
    copy examples\configs\demo_cloud_config.json config.json
)

echo.
echo 4. Installing Python dependencies...
pip install -r requirements.txt

echo.
echo 5. Installing frontend dependencies...
cd frontend
npm install
cd ..

echo.
echo ✅ Setup complete!
echo.
echo To run the demo:
echo.
echo Terminal 1: python -m rag_engine serve
echo Terminal 2: cd frontend && npm run dev
echo.
echo Then visit: http://localhost:3001
echo.
pause
