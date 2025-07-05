"""
Demonstration of the modular orchestration layer.
Shows how to use different orchestrators without changing API code.
"""

import tempfile
import json
import os
from rag_engine.config.loader import load_config
from rag_engine.core.orchestration import create_orchestrator, get_global_registry
from rag_engine.interfaces.base_api import BaseAPIServer
import rag_engine.core.component_registry  # Auto-register components
import rag_engine.core.alternative_orchestrators  # Load alternative orchestrators


class DemoAPIServer(BaseAPIServer):
    """Demo API server for testing orchestrators."""
    
    def create_app(self):
        return f"Demo app with {self.orchestrator_type} orchestrator"
    
    def add_routes(self):
        pass
    
    def start_server(self, **kwargs):
        pass


def demo_orchestrator_swapping():
    """Demonstrate swapping orchestrators without changing API code."""
    
    print("🎬 Modular Orchestration Demo")
    print("=" * 60)
    
    # Create test configuration
    config_data = {
        "documents": [{"path": "test_doc.txt", "type": "text"}],
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
            "persist_directory": "./demo_vectors"
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
            "method": "direct"
        }
    }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        print("📋 Available Components:")
        registry = get_global_registry()
        components = registry.list_components()
        for comp_type, comp_list in components.items():
            if comp_list:
                print(f"  {comp_type}: {', '.join(comp_list)}")
        
        print(f"\n🧠 Testing Different Orchestrators:")
        print("-" * 40)
        
        orchestrator_types = ["default", "hybrid", "multimodal"]
        
        for orch_type in orchestrator_types:
            print(f"\n🔧 Using {orch_type} orchestrator...")
            
            # Create API server with different orchestrator
            api_server = DemoAPIServer(config=config, orchestrator_type=orch_type)
            
            # Create orchestrator directly
            orchestrator = create_orchestrator(orch_type, config, registry)
            
            print(f"   ✅ API Server: {api_server.create_app()}")
            print(f"   ✅ Orchestrator: {orchestrator.__class__.__name__}")
            
            status = orchestrator.get_status()
            print(f"   📊 Status: {status['orchestrator']} (built: {status['is_built']})")
            
            # Show unique features of each orchestrator
            if orch_type == "hybrid":
                print(f"   🔍 Retrieval methods: {status.get('retrieval_methods', [])}")
            elif orch_type == "multimodal":
                print(f"   🎨 Supported modalities: {status.get('supported_modalities', [])}")
        
        print(f"\n🎯 Key Benefits Demonstrated:")
        print("  • Same API server code works with different orchestrators")
        print("  • Easy to swap RAG strategies without code changes")
        print("  • Component registry provides unified interface")
        print("  • Configuration-driven orchestrator selection")
        
        print(f"\n🚀 CLI Usage Examples:")
        print("  # Default orchestrator")
        print(f"  python -m rag_engine serve --config {config_path} --orchestrator default")
        print("  # Hybrid retrieval")
        print(f"  python -m rag_engine serve --config {config_path} --orchestrator hybrid")
        print("  # Multi-modal")
        print(f"  python -m rag_engine serve --config {config_path} --orchestrator multimodal")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


if __name__ == "__main__":
    demo_orchestrator_swapping()
