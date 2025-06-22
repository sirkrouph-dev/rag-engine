"""
Test script for the modular orchestration layer.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the rag_engine to the path
sys.path.append(str(Path(__file__).parent))

from rag_engine.core.orchestration import create_orchestrator, get_global_registry, OrchestratorFactory
from rag_engine.config.loader import load_config
import rag_engine.core.component_registry  # Load default components
import rag_engine.core.alternative_orchestrators  # Load alternative orchestrators


def test_orchestrator_creation():
    """Test creating different orchestrators."""
    print("🧪 Testing Orchestrator Creation")
    print("=" * 50)
      # Create a minimal test config
    test_config = {
        "documents": [
            {
                "path": "test_doc.txt",
                "type": "text"
            }
        ],
        "chunking": {
            "method": "fixed_size",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "max_tokens": 100,
            "overlap": 50
        },
        "embedding": {
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "vectorstore": {
            "provider": "chroma",
            "path": "./test_vector_store_orchestrator",
            "persist_directory": "./test_vector_store_orchestrator"
        },
        "retrieval": {
            "method": "similarity",
            "top_k": 3
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7
        },
        "prompting": {
            "template": "default",
            "system_prompt": "You are a helpful assistant."
        },
        "output": {
            "format": "json"
        }
    }
    
    # Write test config to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        config_path = f.name
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Test available orchestrators
        available_orchestrators = OrchestratorFactory.list_orchestrators()
        print(f"📋 Available orchestrators: {available_orchestrators}")
        
        for orchestrator_type in available_orchestrators:
            print(f"\n🧠 Testing {orchestrator_type} orchestrator...")
            
            try:
                # Create orchestrator
                orchestrator = create_orchestrator(
                    orchestrator_type=orchestrator_type,
                    config=config,
                    registry=get_global_registry()
                )
                
                print(f"✅ Created {orchestrator_type} orchestrator")
                print(f"   Class: {orchestrator.__class__.__name__}")
                
                # Get status without building
                status = orchestrator.get_status()
                print(f"   Status: {status}")
                
            except Exception as e:
                print(f"❌ Failed to create {orchestrator_type} orchestrator: {e}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_component_registry():
    """Test the component registry."""
    print("\n🧪 Testing Component Registry")
    print("=" * 50)
    
    registry = get_global_registry()
    
    # List all components
    components = registry.list_components()
    
    for component_type, component_names in components.items():
        if component_names:  # Only show types that have registered components
            print(f"📦 {component_type}: {', '.join(component_names)}")


def test_orchestrator_api_integration():
    """Test orchestrator integration with API layer."""
    print("\n🧪 Testing API Integration")
    print("=" * 50)
    
    try:
        from rag_engine.interfaces.base_api import BaseAPIServer
        
        # Test that BaseAPIServer accepts orchestrator_type
        class TestAPIServer(BaseAPIServer):
            def create_app(self):
                return None
            def add_routes(self):
                pass
            def start_server(self, **kwargs):
                pass
          # Create test config
        test_config = {
            "documents": [],
            "chunking": {
                "method": "fixed_size",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "max_tokens": 100,
                "overlap": 50
            },
            "embedding": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "vectorstore": {
                "provider": "chroma",
                "path": "./test_vector_store",
                "persist_directory": "./test_vector_store"
            },
            "retrieval": {
                "method": "similarity",
                "top_k": 3
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            },
            "prompting": {
                "template": "default",
                "system_prompt": "You are a helpful assistant."
            },
            "output": {
                "format": "json"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f, indent=2)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            # Test different orchestrator types
            for orchestrator_type in ['default', 'hybrid']:
                server = TestAPIServer(config=config, orchestrator_type=orchestrator_type)
                print(f"✅ Created API server with {orchestrator_type} orchestrator")
                print(f"   Orchestrator type: {server.orchestrator_type}")
                
        except Exception as e:
            print(f"❌ API integration test failed: {e}")
        
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
                
    except ImportError as e:
        print(f"❌ Import error: {e}")


if __name__ == "__main__":
    print("🚀 RAG Engine Orchestration Layer Tests")
    print("=" * 60)
    
    test_component_registry()
    test_orchestrator_creation()
    test_orchestrator_api_integration()
    
    print("\n🎉 Tests completed!")
