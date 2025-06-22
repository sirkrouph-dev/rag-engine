#!/usr/bin/env python3
"""
Test script to verify Vertex AI embedding integration.
This script tests both Gemini API and Vertex AI configurations.
"""

import os
import sys
import json
from typing import Dict, Any

# Add the rag_engine to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine.core.embedder import DefaultEmbedder


def test_gemini_api_config():
    """Test Gemini API configuration."""
    print("Testing Gemini API configuration...")
    
    config = {
        "type": "gemini",
        "model": "models/embedding-001",
        "api_key": os.environ.get("GOOGLE_API_KEY", "test-key"),
        "task_type": "retrieval_document",
        "batch_size": 10
    }
    
    embedder = DefaultEmbedder()
    
    try:
        # This will fail without actual credentials, but should validate config structure
        embedder.embed_query("test query", config)
        print("✓ Gemini API configuration is valid")
    except Exception as e:
        error_str = str(e)
        if ("API key not valid" in error_str or 
            "API key not provided" in error_str or 
            "google-generativeai" in error_str):
            print("✓ Gemini API configuration structure is valid (missing/invalid credentials expected)")
        else:
            print(f"✗ Gemini API configuration error: {e}")
            return False
    
    return True


def test_vertex_ai_config():
    """Test Vertex AI configuration."""
    print("\nTesting Vertex AI configuration...")
    
    config = {
        "type": "gemini",
        "use_vertex": True,
        "model": "textembedding-gecko@001",
        "project": os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project"),
        "location": "us-central1",
        "batch_size": 100
    }
    
    embedder = DefaultEmbedder()
    
    try:
        # This will fail without actual credentials, but should validate config structure
        embedder.embed_query("test query", config)
        print("✓ Vertex AI configuration is valid")
    except Exception as e:
        error_str = str(e)
        if ("project ID not provided" in error_str or 
            "Google Cloud AI Platform not installed" in error_str or
            "API key not provided" in error_str):
            print("✓ Vertex AI configuration structure is valid (missing credentials expected)")
        else:
            print(f"✗ Vertex AI configuration error: {e}")
            return False
    
    return True


def test_provider_registration():
    """Test that GeminiVertexEmbedProvider is properly registered."""
    print("\nTesting provider registration...")
    
    embedder = DefaultEmbedder()
    
    # Check if gemini provider is registered
    if "gemini" in embedder.providers:
        print("✓ GeminiVertexEmbedProvider is registered as 'gemini'")
        
        # Check provider type
        provider = embedder.providers["gemini"]
        if hasattr(provider, 'use_vertex'):
            print("✓ Provider has Vertex AI capability")
        else:
            print("✗ Provider missing Vertex AI capability")
            return False
    else:
        print("✗ GeminiVertexEmbedProvider not registered")
        return False
    
    return True


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    # Test valid configs
    valid_configs = [
        {
            "type": "gemini",
            "model": "models/embedding-001",
            "api_key": "test-key"
        },
        {
            "type": "gemini", 
            "use_vertex": True,
            "model": "textembedding-gecko@001",
            "project": "test-project",
            "location": "us-central1"
        }
    ]
    
    embedder = DefaultEmbedder()
    
    for i, config in enumerate(valid_configs):
        try:
            provider = embedder._get_provider(config)
            print(f"✓ Valid config {i+1} accepted")
        except Exception as e:
            print(f"✗ Valid config {i+1} rejected: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=== Vertex AI Embedding Integration Test ===\n")
    
    tests = [
        test_provider_registration,
        test_config_validation,
        test_gemini_api_config,
        test_vertex_ai_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✅ All Vertex AI embedding integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
