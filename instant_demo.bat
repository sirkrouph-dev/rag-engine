@echo off
title RAG Engine - Instant Demo Setup
color 0A

echo.
echo  ⚡⚡⚡ RAG ENGINE - INSTANT DEMO ⚡⚡⚡
echo  ========================================
echo.
echo  Zero-to-demo in 5 minutes!
echo  No Python knowledge required!
echo.
echo  This will:
echo  ✅ Check for Docker (install if needed)
echo  ✅ Download and start RAG Engine
echo  ✅ Open demo in your browser
echo.
pause

echo.
echo 🔍 Checking if Docker is installed...
docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Docker found!
) else (
    echo ❌ Docker not found. Opening Docker download page...
    start https://docker.com/products/docker-desktop
    echo.
    echo Please install Docker Desktop and run this script again.
    echo (It takes about 2 minutes to install)
    pause
    exit
)

echo.
echo 🚀 Starting RAG Engine (this may take 2-3 minutes the first time)...
echo    Pulling images and starting services...
docker-compose -f docker-compose.demo.yml up -d

echo.
echo ⏳ Waiting for services to be ready...
timeout /t 45 /nobreak >nul

echo.
echo 🌐 Opening demo in your browser...
start http://localhost:3001

echo.
echo ✅ DEMO IS READY!
echo.
echo 🎯 Try these demo steps:
echo    1. Toggle dark/light theme (moon/sun icon)
echo    2. Go to Pipeline → Build Pipeline
echo    3. Go to Documents → Upload demo_document.md
echo    4. Go to Chat → Ask "What is RAG?"
echo.
echo 🛑 To stop the demo: docker-compose -f docker-compose.demo.yml down
echo.
pause
