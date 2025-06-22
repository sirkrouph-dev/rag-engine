#!/usr/bin/env python3
"""
Script to test API endpoints across different frameworks.
"""
import requests
import time
import subprocess
import sys
import threading
import os
from pathlib import Path

def test_api_endpoints(port, framework_name):
    """Test API endpoints for a running server."""
    base_url = f"http://127.0.0.1:{port}"
    
    print(f"\n🧪 Testing {framework_name} API on port {port}")
    
    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Health check passed: {response.json()}")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
            if i == max_retries - 1:
                print(f"❌ {framework_name} server failed to start")
                return False
    
    # Test /query endpoint
    try:
        query_data = {"query": "What is a test?", "top_k": 3}
        response = requests.post(f"{base_url}/query", json=query_data, timeout=10)
        print(f"📝 Query endpoint status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"📄 Query response: {result.get('answer', 'No answer')[:100]}...")
    except Exception as e:
        print(f"❌ Query endpoint failed: {e}")
    
    # Test /build endpoint
    try:
        response = requests.post(f"{base_url}/build", timeout=10)
        print(f"🔨 Build endpoint status: {response.status_code}")
    except Exception as e:
        print(f"❌ Build endpoint failed: {e}")
    
    return True

def start_server(config_path, framework, port):
    """Start a server in a subprocess."""
    cmd = [
        sys.executable, "-m", "rag_engine", "serve",
        "--config", config_path,
        "--framework", framework,
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    
    print(f"🚀 Starting {framework} server: {' '.join(cmd)}")
    
    # Start server and return process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd()
    )
    return process

def main():
    """Main test function."""
    # Use a test config file
    config_path = "test_config_simple.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure you have a test config file.")
        return
    
    frameworks = [
        ("fastapi", 8001),
        ("flask", 8002),
        # ("django", 8003)  # Skip Django for now as it needs setup
    ]
    
    processes = []
    
    try:
        for framework, port in frameworks:
            print(f"\n🔧 Testing {framework.upper()} framework")
            
            # Start server
            process = start_server(config_path, framework, port)
            processes.append(process)
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Test the endpoints
            test_api_endpoints(port, framework)
            
            # Stop this server before testing the next
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            time.sleep(1)
    
    finally:
        # Clean up any remaining processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=3)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        print("\n✅ API framework testing completed!")

if __name__ == "__main__":
    main()
