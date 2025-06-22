"""
Simple test for orchestrator creation.
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


def simple_test():
    """Simple test of orchestrator creation."""
    print("🧪 Simple Orchestrator Test")
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
            "method": "direct"
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


if __name__ == "__main__":
    simple_test()
