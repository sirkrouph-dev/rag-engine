@echo off
echo 🚀 RAG Engine Quick Demo Setup
echo ================================

echo.
echo Choose your setup method:
echo.
echo 1. Docker (Easiest - No Python needed!)
echo 2. Manual (Requires Python)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" goto docker_setup
if "%choice%"=="2" goto manual_setup

:docker_setup
echo.
echo 🐳 Docker Setup Selected
echo ========================

echo.
echo Checking if Docker is installed...
docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Docker found!
    echo.
    echo Starting RAG Engine with Docker...
    echo This will download and start everything automatically.
    echo.
    echo Please wait, this may take a few minutes on first run...
    docker-compose -f docker-compose.demo.yml up --build
) else (
    echo ❌ Docker not found!
    echo.
    echo Please install Docker Desktop from: https://docker.com/products/docker-desktop
    echo Or run: winget install Docker.DockerDesktop
    echo.
    echo Then run this script again.
)
goto end

:manual_setup
echo.
echo 🐍 Manual Setup Selected
echo =========================

echo.
echo 1. Checking if Python is installed...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found!
) else (
    echo ❌ Python not found!
    echo Please install Python from: https://python.org
    echo Then run this script again.
    goto end
)

echo.
echo 2. Checking if Ollama is installed...
where ollama >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama found!
    echo.
    echo 3. Pulling small model (llama3.2:1b)...
    ollama pull llama3.2:1b
    echo.
    echo 4. Using local Ollama config...
    copy examples\configs\demo_local_config.json config.json
) else (
    echo ❌ Ollama not found. Using OpenAI config...
    echo Please edit config.json and add your OpenAI API key.
    copy examples\configs\demo_cloud_config.json config.json
)

echo.
echo 5. Installing Python dependencies...
pip install -r requirements.txt

echo.
echo 6. Installing frontend dependencies...
cd frontend
npm install
cd ..

echo.
echo ✅ Manual setup complete!
echo.
echo To run the demo:
echo.
echo Terminal 1: python -m rag_engine serve
echo Terminal 2: cd frontend && npm run dev
echo.
echo Then visit: http://localhost:3001

:end
echo.
pause
