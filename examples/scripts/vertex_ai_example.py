#!/usr/bin/env python3
"""
Example of using Google Vertex AI embeddings with the RAG Engine.

This script demonstrates how to configure and use both Gemini API and Vertex AI 
for text embeddings in your RAG pipeline.
"""

import os
import json
from typing import List

# Example configurations for different Vertex AI setups

# Configuration 1: Using Gemini API (simpler setup)
gemini_config = {
    "documents": [
        {"type": "txt", "path": "./data/documents/"}
    ],
    "chunking": {
        "method": "semantic",
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "embedding": {
        "type": "gemini",
        "model": "models/embedding-001",
        "api_key": "${GOOGLE_API_KEY}",
        "task_type": "retrieval_document",
        "batch_size": 100
    },
    "vectorstore": {
        "type": "chroma",
        "persist_directory": "./vector_store_gemini",
        "collection_name": "gemini_docs"
    },
    "retrieval": {
        "method": "similarity",
        "top_k": 5
    },
    "llm": {
        "type": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "${OPENAI_API_KEY}"
    }
}

# Configuration 2: Using Vertex AI (enterprise setup)
vertex_ai_config = {
    "documents": [
        {"type": "directory", "path": "./data/documents/", "recursive": True}
    ],
    "chunking": {
        "method": "semantic",
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "embedding": {
        "type": "gemini",
        "use_vertex": True,
        "model": "textembedding-gecko@001",
        "project": "${GOOGLE_CLOUD_PROJECT}",
        "location": "us-central1",
        "batch_size": 100
    },
    "vectorstore": {
        "type": "chroma", 
        "persist_directory": "./vector_store_vertex",
        "collection_name": "vertex_docs"
    },
    "retrieval": {
        "method": "similarity",
        "top_k": 5
    },
    "llm": {
        "type": "openai",
        "model": "gpt-4",
        "api_key": "${OPENAI_API_KEY}"
    }
}

def save_config_examples():
    """Save example configurations to files."""
    
    # Save Gemini API config
    with open("example_gemini_config.json", "w") as f:
        json.dump(gemini_config, f, indent=2)
    print("✓ Saved example_gemini_config.json")
    
    # Save Vertex AI config 
    with open("example_vertex_ai_config.json", "w") as f:
        json.dump(vertex_ai_config, f, indent=2)
    print("✓ Saved example_vertex_ai_config.json")


def print_usage_instructions():
    """Print usage instructions for Vertex AI embeddings."""
    
    print("""
🚀 Google Vertex AI Embedding Usage Instructions

⚠️ EXPERIMENTAL FEATURE - Under Active Development ⚠️

Prerequisites:
1. Install required packages:
   pip install google-generativeai google-cloud-aiplatform

2. Set up authentication:

   For Gemini API:
   export GOOGLE_API_KEY="your_api_key_here"
   
   For Vertex AI (multiple options):
   
   Option A - Service Account File:
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   
   Option B - gcloud CLI:
   gcloud auth application-default login
   
   Option C - Direct config (see examples below)

📁 Configuration Options:

1. Gemini API (Simpler setup):
   {
     "embedding": {
       "type": "gemini",
       "model": "models/embedding-001",
       "api_key": "${{GOOGLE_API_KEY}}",
       "task_type": "retrieval_document"
     }
   }

2. Vertex AI (Enterprise):
   {
     "embedding": {
       "type": "gemini", 
       "use_vertex": true,
       "model": "textembedding-gecko@001",
       "project": "${{GOOGLE_CLOUD_PROJECT}}",
       "location": "us-central1"
     }
   }

3. Vertex AI with Service Account:
   {
     "embedding": {
       "type": "gemini",
       "use_vertex": true,
       "model": "textembedding-gecko@001", 
       "project": "your-project-id",
       "location": "us-central1",
       "credentials_path": "/path/to/service-account.json"
     }
   }

🔧 Available Models:

Gemini API:
- models/embedding-001 (Latest Gemini embedding model)

Vertex AI:
- textembedding-gecko@001 (768 dimensions)
- textembedding-gecko@003 (768 dimensions, latest)

🚀 CLI Usage:

# Using Gemini API
python -m rag_engine serve --config example_gemini_config.json

# Using Vertex AI  
python -m rag_engine serve --config example_vertex_ai_config.json

# Build pipeline with Vertex AI
python -m rag_engine build --config example_vertex_ai_config.json

# Interactive chat
python -m rag_engine chat --config example_vertex_ai_config.json

🔍 Testing Your Setup:

python test_vertex_ai_embedding_fixed.py

This will verify that Vertex AI integration is working correctly.
""")


def main():
    """Main function to demonstrate Vertex AI usage."""
    print("=== Google Vertex AI Embedding Example ===")
    
    save_config_examples()
    print_usage_instructions()
    
    print("""
✅ Example configurations created!

Next steps:
1. Set your API keys/credentials
2. Update file paths in the configs  
3. Run: python -m rag_engine serve --config example_vertex_ai_config.json
""")


if __name__ == "__main__":
    main()
