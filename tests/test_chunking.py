#!/usr/bin/env python3

import sys
sys.path.append('.')

from rag_engine.core.chunker import DefaultChunker

def test_chunking():
    print("Testing chunking...")
    
    # Test document
    document = {
        "content": "This is a test document. It has multiple sentences. We want to chunk it properly.",
        "path": "test.txt",
        "type": "txt"
    }
    
    # Test config
    config = {
        "method": "fixed",
        "chunk_size": 50,
        "chunk_overlap": 10
    }
    
    # Create chunker
    chunker = DefaultChunker()
    
    try:
        chunks = chunker.chunk([document], config)
        print(f"✅ Chunking successful! Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: '{chunk['content'][:30]}...'")
            
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunking()
