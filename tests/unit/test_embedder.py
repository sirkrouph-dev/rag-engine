"""
Unit tests for embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag_engine.core.embedder import HuggingFaceEmbedProvider, OpenAIEmbedProvider, get_embedder


class TestHuggingFaceEmbedProvider:
    """Test HuggingFace embedding functionality."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_query(self, mock_transformer):
        """Test single query embedding."""
        # Mock the SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedProvider()
        config = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        
        result = embedder.embed_query("test query", config)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(["test query"], normalize_embeddings=True)

    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_documents(self, mock_transformer):
        """Test batch document embedding."""
        # Mock the SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedProvider()
        config = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        
        documents = ["doc1", "doc2"]
        result = embedder.embed_documents(documents, config)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch('sentence_transformers.SentenceTransformer')
    def test_different_model_loading(self, mock_transformer):
        """Test that different models are loaded correctly."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedProvider()
        config = {"model": "custom-model"}
        
        embedder.embed_query("test", config)
        
        mock_transformer.assert_called_with("custom-model")


class TestOpenAIEmbedProvider:
    """Test OpenAI embedding functionality."""

    @patch('openai.OpenAI')
    def test_embed_query(self, mock_openai_class):
        """Test single query embedding with OpenAI."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        
        embedder = OpenAIEmbedProvider()
        config = {
            "model": "text-embedding-ada-002",
            "api_key": "test-key"
        }
        
        result = embedder.embed_query("test query", config)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="test query",
            encoding_format="float"
        )

    @patch('openai.OpenAI')
    def test_embed_documents(self, mock_openai_class):
        """Test batch document embedding with OpenAI."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(), MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_response.data[1].embedding = [0.4, 0.5, 0.6]
        mock_client.embeddings.create.return_value = mock_response
        
        embedder = OpenAIEmbedProvider()
        config = {
            "model": "text-embedding-ada-002",
            "api_key": "test-key"
        }
        
        documents = ["doc1", "doc2"]
        result = embedder.embed_documents(documents, config)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=documents,
            encoding_format="float"
        )

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    @patch('openai.OpenAI')
    def test_api_key_from_environment(self, mock_openai_class):
        """Test that API key can be loaded from environment."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        
        embedder = OpenAIEmbedProvider()
        config = {"model": "text-embedding-ada-002"}  # No API key in config
        
        result = embedder.embed_query("test", config)
        
        assert result == [0.1, 0.2, 0.3]
        mock_openai_class.assert_called_with(api_key="env-key")


class TestEmbedderFactory:
    """Test the embedder factory function."""

    def test_get_huggingface_embedder(self):
        """Test getting HuggingFace embedder from factory."""
        config = {"provider": "huggingface"}
        embedder = get_embedder(config)
        # The factory returns a DefaultEmbedder, so we check its provider
        assert hasattr(embedder, 'providers')
        provider = embedder._get_provider(config)
        assert isinstance(provider, HuggingFaceEmbedProvider)

    def test_get_openai_embedder(self):
        """Test getting OpenAI embedder from factory."""
        config = {"provider": "openai"}
        embedder = get_embedder(config)
        # The factory returns a DefaultEmbedder, so we check its provider
        assert hasattr(embedder, 'providers')
        provider = embedder._get_provider(config)
        assert isinstance(provider, OpenAIEmbedProvider)

    def test_get_default_embedder(self):
        """Test that default embedder is created."""
        config = {}
        embedder = get_embedder(config)
        # Should return a DefaultEmbedder instance
        assert hasattr(embedder, 'providers')

    def test_get_embedder_invalid_provider(self):
        """Test that invalid provider raises error during usage."""
        config = {"provider": "invalid"}
        embedder = get_embedder(config)
        # Error should happen when trying to use the embedder
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            embedder.embed_query("test", config)
