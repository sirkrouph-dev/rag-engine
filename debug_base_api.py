#!/usr/bin/env python3
"""
Test script to debug base_api.py import issues.
"""
import sys
import traceback

print("Testing imports step by step...")

try:
    print("1. Testing basic imports...")
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, List
    print("   ✅ Basic imports OK")
except Exception as e:
    print(f"   ❌ Basic imports failed: {e}")
    sys.exit(1)

try:
    print("2. Testing RAG engine imports...")
    from rag_engine.core.pipeline import Pipeline
    print("   ✅ Pipeline import OK")
except Exception as e:
    print(f"   ❌ Pipeline import failed: {e}")
    traceback.print_exc()

try:
    from rag_engine.config.schema import RAGConfig
    print("   ✅ RAGConfig import OK")
except Exception as e:
    print(f"   ❌ RAGConfig import failed: {e}")
    traceback.print_exc()

try:
    from rag_engine.config.loader import load_config
    print("   ✅ load_config import OK")
except Exception as e:
    print(f"   ❌ load_config import failed: {e}")
    traceback.print_exc()

print("3. Testing base_api.py content...")
try:
    # Read and execute the file content
    with open('rag_engine/interfaces/base_api.py', 'r') as f:
        content = f.read()
    
    print(f"   File size: {len(content)} bytes")
    print(f"   Lines: {len(content.splitlines())}")
    
    # Try to exec the content
    exec(content)
    print("   ✅ File execution OK")
    
    # Check if classes were defined
    if 'BaseAPIServer' in locals():
        print("   ✅ BaseAPIServer defined")
    else:
        print("   ❌ BaseAPIServer not defined")
    
    if 'APIModelFactory' in locals():
        print("   ✅ APIModelFactory defined")
    else:
        print("   ❌ APIModelFactory not defined")
        
except Exception as e:
    print(f"   ❌ File execution failed: {e}")
    traceback.print_exc()

print("4. Testing module import...")
try:
    import rag_engine.interfaces.base_api
    print("   ✅ Module import OK")
    print(f"   Module attributes: {[attr for attr in dir(rag_engine.interfaces.base_api) if not attr.startswith('_')]}")
except Exception as e:
    print(f"   ❌ Module import failed: {e}")
    traceback.print_exc()
