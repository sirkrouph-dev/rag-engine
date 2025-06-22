#!/usr/bin/env python3
"""
Simple FastAPI server test with better error handling.
"""
import uvicorn
from fastapi import FastAPI
import sys
import traceback

def test_simple_fastapi():
    """Test a simple FastAPI server startup."""
    try:
        app = FastAPI(title="Test RAG API")
        
        @app.get("/")
        def root():
            return {"message": "Hello World"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy"}
        
        print("🚀 Starting simple FastAPI server on port 8001...")
        print("📋 Available endpoints:")
        print("   - http://127.0.0.1:8001/")
        print("   - http://127.0.0.1:8001/health")
        print("   - http://127.0.0.1:8001/docs")
        print("\n⚠️  Press Ctrl+C to stop")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ FastAPI server failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_fastapi()
