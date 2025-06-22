#!/usr/bin/env python3
"""
Direct test of the serve functionality without CLI.
"""
import sys
import traceback
import signal
import threading
import time

def test_serve_direct():
    """Test the serve functionality directly."""
    try:
        print("🧪 Testing serve functionality directly...")
        
        # Import the components
        from rag_engine.interfaces.base_api import APIModelFactory
        from rag_engine.interfaces.api import FastAPIServer
        
        # Create server
        print("🔧 Creating server...")
        server = APIModelFactory.create_server("fastapi", config_path="test_config_simple.json")
        
        # Create app (this should be safe)
        print("🔧 Creating FastAPI app...")
        app = server.create_app()
        
        print("🚀 Starting server with timeout...")
        
        # Set up a timeout to prevent infinite hanging
        def timeout_handler():
            print("\n⏰ Timeout reached - stopping test")
            import os
            os._exit(1)
        
        # Start timeout timer
        timer = threading.Timer(10.0, timeout_handler)  # 10 second timeout
        timer.start()
        
        try:
            # This is where it might hang
            print("🔥 Calling start_server...")
            server.start_server(host="127.0.0.1", port=8002, reload=False)
        except KeyboardInterrupt:
            print("\n⌨️  Interrupted by user")
        finally:
            timer.cancel()
            
    except Exception as e:
        print(f"❌ Direct serve test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_serve_direct()
