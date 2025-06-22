#!/usr/bin/env python3
"""
Simple test to verify the pipeline works with our config.
"""
import sys
import traceback
from rag_engine.config.loader import load_config
from rag_engine.core.pipeline import Pipeline

def test_pipeline():
    """Test pipeline initialization and basic operations."""
    try:
        print("🔧 Loading config...")
        config = load_config("test_config_simple.json")
        print(f"✅ Config loaded: {config.embedding.provider}, {config.vectorstore.provider}, {config.llm.provider}")
        
        print("🏗️  Creating pipeline...")
        pipeline = Pipeline(config)
        print("✅ Pipeline created")
        
        print("📚 Building pipeline (this may take a moment)...")
        try:
            pipeline.build()
            print("✅ Pipeline built successfully!")
            return True
        except Exception as e:
            print(f"⚠️  Pipeline build failed: {e}")
            # This is expected if we don't have proper API keys or models
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pipeline()
