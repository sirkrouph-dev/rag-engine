"""
Unit tests for document chunking functionality.
"""
import pytest
from rag_engine.core.chunker import FixedSizeChunker, get_chunker


class TestFixedSizeChunker:
    """Test the FixedSizeChunker implementation."""

    def test_basic_chunking(self, sample_document):
        """Test basic document chunking."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 50,
            "overlap": 10
        }
        
        chunks = chunker.chunk(sample_document, config)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_chunk_size_limits(self):
        """Test that chunks respect the max_tokens limit."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 20,
            "overlap": 5
        }
        
        # Create a document with known content
        document = {
            "content": "This is a very long document with many words that should be split into multiple chunks based on the token limit we have set.",
            "id": "test_doc"
        }
        
        chunks = chunker.chunk(document, config)
        
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Each chunk should be within the token limit (roughly)
        for chunk in chunks:
            # Rough word count check (tokens ≈ words for simple text)
            word_count = len(chunk["content"].split())
            assert word_count <= config["max_tokens"] + 5  # Allow some flexibility

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 15,
            "overlap": 5
        }
        
        document = {
            "content": "First sentence here. Second sentence follows. Third sentence continues. Fourth sentence ends.",
            "id": "test_doc"
        }
        
        chunks = chunker.chunk(document, config)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            first_chunk_words = chunks[0]["content"].split()
            second_chunk_words = chunks[1]["content"].split()
            
            # There should be some overlap between chunks
            overlap_found = any(word in second_chunk_words for word in first_chunk_words[-5:])
            assert overlap_found, "No overlap found between consecutive chunks"

    def test_short_document_single_chunk(self):
        """Test that short documents create only one chunk."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 100,
            "overlap": 10
        }
        
        document = {
            "content": "Short document.",
            "id": "test_doc"
        }
        
        chunks = chunker.chunk(document, config)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short document."

    def test_empty_document(self):
        """Test handling of empty documents."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 50,
            "overlap": 10
        }
        
        document = {
            "content": "",
            "id": "empty_doc"
        }
        
        chunks = chunker.chunk(document, config)
          # Should return empty list or single empty chunk
        assert len(chunks) <= 1
        if chunks:
            assert chunks[0]["content"] == ""

    def test_chunk_metadata_preservation(self, sample_document):
        """Test that chunk metadata includes document information."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 30,
            "overlap": 5
        }
        
        chunks = chunker.chunk(sample_document, config)
        
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            assert "doc_type" in metadata
            assert metadata["doc_type"] == sample_document["type"]
            assert "start_char" in metadata
            assert "end_char" in metadata
            assert "chunk_method" in metadata
            assert metadata["chunk_method"] == "fixed_size"

    def test_zero_overlap_configuration(self):
        """Test chunking with zero overlap."""
        chunker = FixedSizeChunker()
        config = {
            "max_tokens": 10,
            "overlap": 0
        }
        
        document = {
            "content": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.",
            "id": "test_doc"
        }
        
        chunks = chunker.chunk(document, config)
        
        assert len(chunks) > 1
        
        # With zero overlap, no content should be repeated
        all_content = " ".join(chunk["content"] for chunk in chunks)
        original_words = document["content"].split()
        result_words = all_content.split()
        
        # Should have roughly the same number of words (allowing for chunking boundaries)
        assert abs(len(original_words) - len(result_words)) <= 2


class TestChunkerFactory:
    """Test the chunker factory function."""

    def test_get_fixed_chunker(self):
        """Test getting a chunker."""
        config = {"method": "fixed"}
        chunker = get_chunker(config)
        
        from rag_engine.core.chunker import DefaultChunker
        assert isinstance(chunker, DefaultChunker)

    def test_get_chunker_default(self):
        """Test getting default chunker when method not specified."""
        config = {}
        chunker = get_chunker(config)
        
        from rag_engine.core.chunker import DefaultChunker
        assert isinstance(chunker, DefaultChunker)

    def test_get_chunker_invalid_method(self):
        """Test error handling for invalid chunker method."""
        config = {"method": "nonexistent_method"}
        chunker = get_chunker(config)
        
        # The DefaultChunker should raise the error when chunk() is called
        with pytest.raises(ValueError, match="Unsupported chunking method"):
            chunker.chunk([{"content": "test"}], config)
