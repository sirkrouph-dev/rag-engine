# Chunker interface and implementations

import re
import nltk
from typing import List, Dict, Any, Union
from abc import ABC
from rag_engine.core.base import BaseChunker

class ChunkStrategy(ABC):
    """Base abstract class for different chunking strategies."""
    def chunk(self, document: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class FixedSizeChunker(ChunkStrategy):
    """Chunk by fixed character length with optional overlap."""
    def chunk(self, document: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = document["content"]
        chunk_size = config.get("chunk_size", 1000)  # Default 1000 characters
        chunk_overlap = config.get("chunk_overlap", 0)  # Default no overlap
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be non-negative and less than chunk size")
            
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            if end >= len(content):
                end = len(content)
            
            chunk_text = content[start:end]
            chunks.append({
                "content": chunk_text,
                "path": document.get("path", ""),
                "metadata": {
                    "doc_type": document.get("type", "unknown"),
                    "start_char": start,
                    "end_char": end,
                    "chunk_method": "fixed_size"
                }
            })
            
            start = end - chunk_overlap if chunk_overlap > 0 else end
            
        return chunks


class SentenceChunker(ChunkStrategy):
    """Chunk by sentences with configurable max chunk size in sentences."""
    def chunk(self, document: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        content = document["content"]
        sentences = nltk.sent_tokenize(content)
        max_sentences = config.get("max_sentences", 5)  # Default 5 sentences per chunk
        sentence_overlap = config.get("sentence_overlap", 0)  # Default no overlap
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            end_idx = min(i + max_sentences, len(sentences))
            chunk_text = " ".join(sentences[i:end_idx])
            
            chunks.append({
                "content": chunk_text,
                "path": document.get("path", ""),
                "metadata": {
                    "doc_type": document.get("type", "unknown"),
                    "start_sentence": i,
                    "end_sentence": end_idx,
                    "chunk_method": "sentence"
                }
            })
            
            i = end_idx - sentence_overlap if sentence_overlap > 0 else end_idx
            
        return chunks


class SemanticChunker(ChunkStrategy):
    """Chunk by semantic elements like headings, paragraphs, etc."""
    def chunk(self, document: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        # This works differently based on document type
        if document.get("type") == "html":
            # HTML documents with element-based chunks are already prepared in the loader
            if isinstance(document.get("content"), list):
                # HTML loader already chunked by elements
                chunks = []
                for elem in document["content"]:
                    chunks.append({
                        "content": elem["content"],
                        "path": document.get("path", ""),
                        "metadata": {
                            "doc_type": "html",
                            "element_type": elem["type"],
                            "chunk_method": "semantic"
                        }
                    })
                return chunks
        
        # For text-based documents, use paragraph separation
        content = document["content"]
        paragraphs = re.split(r'\n\s*\n', content)  # Split by blank lines
        
        chunks = []
        for i, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                chunks.append({
                    "content": para.strip(),
                    "path": document.get("path", ""),
                    "metadata": {
                        "doc_type": document.get("type", "unknown"),
                        "paragraph_idx": i,
                        "chunk_method": "semantic"
                    }
                })
                
        return chunks


class RecursiveChunker(ChunkStrategy):
    """Advanced recursive chunking strategy for nested documents."""
    def chunk(self, document: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = document["content"]
        chunk_size = config.get("chunk_size", 1000)
        chunk_overlap = config.get("chunk_overlap", 100)
        separators = config.get("separators", ["\n\n", "\n", ". ", " ", ""])
        
        return self._recursive_split(content, document, separators, chunk_size, chunk_overlap)
        
    def _recursive_split(self, text, document, separators, chunk_size, chunk_overlap, depth=0):
        """Recursively split text using different separators."""
        if not text or depth >= len(separators):
            return []
            
        separator = separators[depth]
        parts = text.split(separator) if separator else list(text)
        
        # If parts are small enough, return them directly
        if all(len(part) <= chunk_size for part in parts):
            return [{
                "content": part,
                "path": document.get("path", ""),
                "metadata": {
                    "doc_type": document.get("type", "unknown"),
                    "chunk_method": "recursive",
                    "depth": depth,
                    "separator": separator
                }
            } for part in parts if part.strip()]
            
        # Otherwise, recursively split the parts
        chunks = []
        for part in parts:
            if len(part) <= chunk_size:
                if part.strip():
                    chunks.append({
                        "content": part,
                        "path": document.get("path", ""),
                        "metadata": {
                            "doc_type": document.get("type", "unknown"),
                            "chunk_method": "recursive",
                            "depth": depth,
                            "separator": separator
                        }
                    })
            else:
                chunks.extend(self._recursive_split(
                    part, document, separators, chunk_size, chunk_overlap, depth + 1
                ))
        
        return chunks


class DefaultChunker(BaseChunker):
    """Main chunker that delegates to appropriate strategy based on config."""
    def __init__(self):
        self.strategies = {
            "fixed": FixedSizeChunker(),
            "sentence": SentenceChunker(),
            "semantic": SemanticChunker(),
            "recursive": RecursiveChunker()
        }
    
    def chunk(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        method = config.get("method", "fixed").lower()
        if method not in self.strategies:
            raise ValueError(f"Unsupported chunking method: {method}")
        
        strategy = self.strategies[method]
        chunks = []
        
        for doc in documents:
            doc_chunks = strategy.chunk(doc, config)
            chunks.extend(doc_chunks)
            
        return chunks


# Factory function to get the appropriate chunker
def get_chunker(config: Dict[str, Any]) -> BaseChunker:
    return DefaultChunker()
