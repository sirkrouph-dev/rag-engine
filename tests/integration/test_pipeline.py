"""
Integration tests for the RAG pipeline.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import patch, Mock
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.loader import load_config
from rag_engine.config.schema import RAGConfig


class TestPipelineIntegration:
    """Test end-to-end pipeline functionality."""

    @pytest.fixture
    def pipeline_config(self, temp_dir):
        """Create a complete pipeline configuration for testing."""
        # Create a test document
        test_doc_path = os.path.join(temp_dir, "test_document.txt")
        with open(test_doc_path, "w", encoding="utf-8") as f:
            f.write("This is a test document for the RAG pipeline. "
                   "It contains multiple sentences to test the chunking functionality. "
                   "The pipeline should be able to process this document successfully.")
        
        config = {
            "documents": [
                {"type": "txt", "path": test_doc_path}
            ],
            "chunking": {
                "method": "fixed",
                "max_tokens": 50,
                "overlap": 10
            },
            "embedding": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "vectorstore": {
                "provider": "chroma",
                "persist_directory": os.path.join(temp_dir, "vector_store"),
                "collection_name": "test_collection"
            },
            "retrieval": {
                "top_k": 3
            },
            "prompting": {
                "system_prompt": "You are a helpful assistant. Answer based on the provided context."
            },            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "api_key": "test_api_key"
            },
            "output": {
                "method": "text"
            }
        }
        
        return config

    def test_pipeline_initialization(self, pipeline_config):
        """Test that pipeline initializes with valid configuration."""
        config = RAGConfig(**pipeline_config)
        pipeline = Pipeline(config)
        
        assert pipeline is not None
        assert isinstance(pipeline.config, RAGConfig)

    @pytest.mark.integration
    def test_pipeline_build_mocked_embedding(self, pipeline_config):
        """Test pipeline build process with mocked embedding."""
        config = RAGConfig(**pipeline_config)
        pipeline = Pipeline(config)        # Mock the embedder to avoid network calls
        with patch('rag_engine.core.embedder.HuggingFaceEmbedProvider.embed_query') as mock_embed:
            with patch('rag_engine.core.embedder.HuggingFaceEmbedProvider.embed_documents') as mock_embed_batch:
                # Mock embedding responses - dynamic based on input length
                mock_embed.return_value = [0.1] * 384  # all-MiniLM-L6-v2 dimension
                def mock_embed_docs(texts, config):
                    return [[0.1] * 384 for _ in texts]  # Return one embedding per text
                mock_embed_batch.side_effect = mock_embed_docs
                
                # Mock ChromaDB to avoid actual database operations
                with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.add') as mock_add:
                    with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.persist') as mock_persist:
                          # Run the build process
                        result = pipeline.build()
                        
                        # Verify the build completed successfully
                        assert result is None  # build() returns None when successful
                        
                        # Verify that embedding and storage methods were called
                        assert mock_embed_batch.called or mock_embed.called
                        mock_add.assert_called_once()
                        mock_persist.assert_called_once()

    @pytest.mark.integration
    def test_pipeline_build_with_multiple_documents(self, temp_dir):
        """Test pipeline with multiple documents."""
        # Create multiple test documents
        doc1_path = os.path.join(temp_dir, "doc1.txt")
        doc2_path = os.path.join(temp_dir, "doc2.txt")
        
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("First document content for testing.")
        
        with open(doc2_path, "w", encoding="utf-8") as f:
            f.write("Second document with different content for testing the pipeline.")
        
        config = {
            "documents": [
                {"type": "txt", "path": doc1_path},
                {"type": "txt", "path": doc2_path}
            ],
            "chunking": {
                "method": "fixed",
                "max_tokens": 30,
                "overlap": 5
            },
            "embedding": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },            "vectorstore": {
                "provider": "chroma",
                "persist_directory": os.path.join(temp_dir, "vector_store")
            },
            "retrieval": {"top_k": 5},
            "prompting": {"system_prompt": "You are a helpful assistant."},
            "llm": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
            "output": {"method": "text"}
        }
        
        config = RAGConfig(**config)
        pipeline = Pipeline(config)
        
        # Mock embedding and storage
        with patch('rag_engine.core.embedder.HuggingFaceEmbedProvider.embed_documents') as mock_embed_batch:
            with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.add') as mock_add:
                with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.persist') as mock_persist:
                    
                    # Dynamic mock to return correct number of embeddings
                    def mock_embed_docs(texts, config):
                        return [[0.1] * 384 for _ in texts]  # Return one embedding per text
                    mock_embed_batch.side_effect = mock_embed_docs
                    
                    result = pipeline.build()
                    
                    # Verify multiple documents were processed
                    mock_add.assert_called_once()
                    mock_persist.assert_called_once()

    @pytest.mark.integration
    def test_pipeline_query_mocked_llm(self, pipeline_config):
        """Test pipeline query functionality with mocked LLM."""
        config = RAGConfig(**pipeline_config)
        pipeline = Pipeline(config)
        
        # Mock all dependencies including vectorstore loading
        with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.load') as mock_load:
            with patch('rag_engine.core.embedder.HuggingFaceEmbedProvider.embed_query') as mock_embed:
                with patch('rag_engine.core.llm.OpenAIProvider.generate') as mock_llm:
                    
                    # Mock vectorstore loading (so chat doesn't fail immediately)
                    mock_load.return_value = None
                    
                    # Mock embedding for query
                    mock_embed.return_value = [0.1] * 384
                    
                    # Mock LLM response
                    mock_llm.return_value = "This is a generated response based on the context."
                    
                    # Test query
                    response = pipeline.chat("What is in the document?")
                    
                    # Verify the response
                    assert isinstance(response, str)
                    assert len(response) > 0
                    
                    # Verify key components were called (retrieval might fail but embedder should be called)
                    mock_embed.assert_called_once()
                    mock_llm.assert_called_once()
                    # Note: vectorstore query might not be called due to empty index, which is expected in this test

    def test_pipeline_error_handling_missing_document(self, temp_dir):
        """Test pipeline error handling for missing documents."""
        from rag_engine.config.schema import RAGConfig
        
        config_dict = {
            "documents": [
                {"type": "txt", "path": "/nonexistent/document.txt"}
            ],
            "chunking": {"method": "fixed", "max_tokens": 100, "overlap": 10},
            "embedding": {"provider": "huggingface", "model": "sentence-transformers/all-MiniLM-L6-v2"},
            "vectorstore": {"provider": "chroma", "persist_directory": "./test_vector_store"},
            "retrieval": {"top_k": 5},
            "prompting": {"system_prompt": "You are a helpful assistant."},
            "llm": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
            "output": {"method": "text"}
        }
        
        config = RAGConfig(**config_dict)
        pipeline = Pipeline(config)
        
        # The pipeline should handle missing documents gracefully
        # Let's test that it doesn't crash and can be built with 0 chunks
        with patch('rag_engine.core.embedder.HuggingFaceEmbedProvider.embed_documents') as mock_embed:
            with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.add') as mock_add:
                with patch('rag_engine.core.vectorstore.ChromaDBVectorStore.persist') as mock_persist:
                    def mock_embed_docs(texts, config):
                        return [[0.1] * 384 for _ in texts]  # Return one embedding per text
                    mock_embed.side_effect = mock_embed_docs
                      # Should not raise an error, just process 0 chunks
                    result = pipeline.build()
                    assert result is None  # build() returns None when successful

    def test_pipeline_with_config_file(self, temp_dir):
        """Test pipeline initialization from config file."""        # Create config file
        config_data = {
            "documents": [{"type": "txt", "path": "./test.txt"}],
            "chunking": {"method": "fixed", "max_tokens": 100, "overlap": 10},
            "embedding": {"provider": "huggingface", "model": "test-model"},
            "vectorstore": {"provider": "chroma", "persist_directory": "./test_vector_store"},
            "retrieval": {"top_k": 5},
            "prompting": {"system_prompt": "You are a helpful assistant."},
            "llm": {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.7},
            "output": {"method": "text"}
        }
        
        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        # Load config and create pipeline
        config = load_config(config_path)
        pipeline = Pipeline(config)
        assert isinstance(pipeline.config, RAGConfig)
        assert pipeline.config.documents[0].path == "./test.txt"
    
    @pytest.mark.integration
    def test_pipeline_component_integration(self, pipeline_config):
        """Test that all pipeline components work together."""
        config = RAGConfig(**pipeline_config)
        pipeline = Pipeline(config)
        
        # Test that we can access all required components
        assert hasattr(pipeline, '_load_documents')
        assert hasattr(pipeline, '_chunk_documents')
        assert hasattr(pipeline, '_embed_and_store')
        assert hasattr(pipeline, '_generate_response')
        assert hasattr(pipeline, 'build')
        assert hasattr(pipeline, 'chat')
        
        # Test component instantiation doesn't fail
        from rag_engine.core.loader import LOADER_REGISTRY
        from rag_engine.core.chunker import get_chunker
        from rag_engine.core.embedder import get_embedder
        from rag_engine.core.vectorstore import get_vector_store
        from rag_engine.core.retriever import get_retriever
        from rag_engine.core.llm import get_llm
        
        # Verify all components can be instantiated
        chunker = get_chunker(pipeline_config.get("chunking", {}))
        embedder = get_embedder(pipeline_config.get("embedding", {}))
        vectorstore = get_vector_store(pipeline_config.get("vectorstore", {}))
        retriever = get_retriever(pipeline_config.get("retrieval", {}))
        llm = get_llm(pipeline_config.get("llm", {}))
        
        assert chunker is not None
        assert embedder is not None
        assert vectorstore is not None
        assert retriever is not None
        assert llm is not None
