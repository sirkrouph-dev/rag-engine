#!/usr/bin/env python3
"""
Test the RAG Engine FastAPI server creation step by step.
"""
import sys
import traceback

def test_rag_api_creation():
    """Test RAG API server creation step by step."""
    try:
        print("1. Testing base API import...")
        from rag_engine.interfaces.base_api import APIModelFactory, BaseAPIServer
        print("   ✅ Base API imported")
        
        print("2. Testing FastAPI server import...")
        from rag_engine.interfaces.api import FastAPIServer
        print("   ✅ FastAPI server imported")
        
        print("3. Checking registered frameworks...")
        frameworks = APIModelFactory.list_frameworks()
        print(f"   📋 Available: {frameworks}")
        
        print("4. Testing server creation...")
        server = APIModelFactory.create_server("fastapi", config_path="test_config_simple.json")
        print("   ✅ Server created")
        
        print("5. Testing app creation...")
        app = server.create_app()
        print(f"   ✅ FastAPI app created: {type(app)}")
        
        print("6. Testing manual server start (non-blocking)...")
        print("   ⚠️  This should work without hanging...")
        
        # Instead of calling start_server (which blocks), let's test uvicorn directly
        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="info")
        server_instance = uvicorn.Server(config)
        print("   ✅ Uvicorn server configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed at step: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag_api_creation()
