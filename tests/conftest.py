# Test fixtures and shared utilities
import os
import pytest
import tempfile
import json
import shutil
import time
import gc
from pathlib import Path
from typing import Dict, Any

def safe_cleanup(path):
    """Safely cleanup ChromaDB files on Windows."""
    if not os.path.exists(path):
        return
    
    # Force garbage collection to release any handles
    gc.collect()
    
    # Wait a bit for ChromaDB to release file handles
    time.sleep(0.1)
    
    # Try multiple times to delete the directory
    for i in range(3):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
            break
        except (PermissionError, OSError):
            time.sleep(0.2)
            gc.collect()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        safe_cleanup(temp_dir)

@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    content = "This is a sample document for testing the RAG engine. It contains multiple sentences to test chunking functionality."
    file_path = os.path.join(temp_dir, "sample.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "documents": [
            {"type": "txt", "path": "./sample.txt"}
        ],
        "chunking": {
            "method": "fixed",
            "max_tokens": 100,
            "overlap": 10
        },
        "embedding": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "vectorstore": {
            "provider": "chroma",
            "persist_directory": "./test_vector_store"
        },
        "retrieval": {
            "top_k": 3
        },
        "prompting": {
            "system_prompt": "You are a helpful assistant."
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "api_key": "${OPENAI_API_KEY}"
        },
        "output": {
            "method": "console"
        }
    }

@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """Create a sample config file."""
    config_path = os.path.join(temp_dir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(sample_config, f, indent=2)
    return config_path

@pytest.fixture
def sample_document():
    """Sample document data for testing."""
    return {
        "id": "test_doc_1",
        "type": "txt",
        "content": "This is a test document with some content for testing purposes.",
        "path": "/test/path/document.txt",
        "metadata": {
            "title": "Test Document",
            "author": "Test Author"
        }
    }

@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "id": "chunk_1",
            "content": "This is the first chunk of content.",
            "metadata": {
                "document_id": "test_doc_1",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 35
            }
        },
        {
            "id": "chunk_2", 
            "content": "This is the second chunk of content.",
            "metadata": {
                "document_id": "test_doc_1",
                "chunk_index": 1,
                "start_char": 25,
                "end_char": 61
            }
        }
    ]

@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 76  # 384 dimensions like all-MiniLM-L6-v2
