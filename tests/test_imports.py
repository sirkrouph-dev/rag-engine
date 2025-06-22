#!/usr/bin/env python3
"""Simple test to check if our pipeline can be imported and initialized."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work."""
    try:
        print("Testing imports...")
        
        # Test config loading
        from rag_engine.config.loader import load_config
        print("✓ Config loader imported")
        
        # Test pipeline import
        from rag_engine.core.pipeline import Pipeline
        print("✓ Pipeline imported")
        
        # Test loading config
        config = load_config('test_config.json')
        print("✓ Config loaded")
        
        # Test creating pipeline
        pipeline = Pipeline(config)
        print("✓ Pipeline created")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
