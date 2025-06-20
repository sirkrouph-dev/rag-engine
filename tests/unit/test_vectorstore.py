"""
Unit tests for vector store functionality.
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from rag_engine.core.vectorstore import ChromaDBVectorStore, get_vector_store, DefaultVectorStore, FAISSVectorStore


class TestChromaDBVectorStore:
    """Test ChromaDB vector store functionality."""

    def test_chromadb_initialization(self):
        """Test that ChromaDB vector store initializes properly."""
        vectorstore = ChromaDBVectorStore()
        assert vectorstore is not None
        assert vectorstore.client is None
        assert vectorstore.collection is None

    def test_get_client_persistent(self):
        """Test creating persistent ChromaDB client."""
        vectorstore = ChromaDBVectorStore()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "persist_directory": temp_dir
            }
            
            with patch.object(vectorstore, 'chromadb') as mock_chromadb:
                mock_client = Mock()
                mock_chromadb.PersistentClient.return_value = mock_client
                
                client = vectorstore._get_client(config)
                
                assert client == mock_client
                mock_chromadb.PersistentClient.assert_called_once_with(path=temp_dir)

    def test_get_client_http(self):
        """Test creating HTTP ChromaDB client."""
        vectorstore = ChromaDBVectorStore()
        config = {
            "host": "localhost",
            "port": 8000
        }
        
        with patch.object(vectorstore, 'chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_chromadb.HttpClient.return_value = mock_client
            
            client = vectorstore._get_client(config)
            
            assert client == mock_client
            mock_chromadb.HttpClient.assert_called_once_with(host="localhost", port=8000)

    def test_get_client_in_memory(self):
        """Test creating in-memory ChromaDB client."""
        vectorstore = ChromaDBVectorStore()
        config = {}  # No persist_directory or host/port
        
        with patch.object(vectorstore, 'chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_chromadb.Client.return_value = mock_client
            
            client = vectorstore._get_client(config)
            
            assert client == mock_client
            mock_chromadb.Client.assert_called_once()

    def test_get_collection_existing(self):
        """Test getting an existing collection."""
        vectorstore = ChromaDBVectorStore()
        config = {
            "collection_name": "test_collection"
        }
        
        with patch.object(vectorstore, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_get_client.return_value = mock_client
            
            collection = vectorstore._get_collection(config)
            
            assert collection == mock_collection
            mock_client.get_collection.assert_called_once_with(name="test_collection")

    def test_get_collection_create_new(self):
        """Test creating a new collection when it doesn't exist."""
        vectorstore = ChromaDBVectorStore()
        config = {
            "collection_name": "new_collection",
            "metric": "cosine"
        }
        
        with patch.object(vectorstore, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_collection = Mock()
            
            # Simulate collection not existing (first call raises exception)
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            mock_get_client.return_value = mock_client
            
            collection = vectorstore._get_collection(config)
            
            assert collection == mock_collection
            mock_client.create_collection.assert_called_once()

    def test_add_documents(self, sample_chunks, mock_embedding):
        """Test adding documents to ChromaDB."""
        vectorstore = ChromaDBVectorStore()
        config = {"collection_name": "test_collection"}
        
        # Prepare documents with embeddings
        documents = []
        for chunk in sample_chunks:
            doc = chunk.copy()
            doc["embedding"] = mock_embedding
            documents.append(doc)
        
        with patch.object(vectorstore, '_get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            vectorstore.add(documents, config)
            
            # Verify collection.add was called
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            
            assert len(call_args["ids"]) == 2
            assert len(call_args["embeddings"]) == 2
            assert len(call_args["metadatas"]) == 2
            assert len(call_args["documents"]) == 2

    def test_add_documents_without_embeddings(self, sample_chunks):
        """Test handling documents without embeddings."""
        vectorstore = ChromaDBVectorStore()
        config = {"collection_name": "test_collection"}
        
        # Documents without embeddings
        documents = sample_chunks.copy()
        
        with patch.object(vectorstore, '_get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            vectorstore.add(documents, config)
            
            # Should not call collection.add since no valid documents
            mock_collection.add.assert_not_called()

    def test_add_empty_documents(self):
        """Test adding empty document list."""
        vectorstore = ChromaDBVectorStore()
        config = {"collection_name": "test_collection"}
        
        with patch.object(vectorstore, '_get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            vectorstore.add([], config)
              # Should not call collection.add for empty list
            mock_collection.add.assert_not_called()

    def test_query_vectorstore(self, mock_embedding):
        """Test querying the vector store."""
        vectorstore = ChromaDBVectorStore()
        config = {
            "collection_name": "test_collection",
            "top_k": 3
        }
        
        with patch.object(vectorstore, '_get_collection') as mock_get_collection:
            mock_collection = Mock()
            
            # Mock query response
            mock_response = {
                "ids": [["doc1", "doc2"]],
                "distances": [[0.1, 0.3]],
                "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
                "documents": [["Content 1", "Content 2"]]
            }
            mock_collection.query.return_value = mock_response
            mock_get_collection.return_value = mock_collection
            
            results = vectorstore.query(mock_embedding, config)
            
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["document"]["content"] == "Content 1"
            assert results[0]["score"] == 1.0 / (1.0 + 0.1)  # 1 / (1 + distance)
            
            mock_collection.query.assert_called_once()

    def test_persist(self):
        """Test persisting the vector store."""
        vectorstore = ChromaDBVectorStore()
        config = {"collection_name": "test_collection"}
        
        with patch.object(vectorstore, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Test that persist calls without error
            vectorstore.persist(config)
            
            # For ChromaDB, persist is a no-op since it auto-persists
            # Just verify no exceptions are raised


class TestVectorStoreFactory:
    """Test the vector store factory function."""

    def test_get_chroma_vectorstore(self):
        """Test getting ChromaDB vector store."""
        config = {"provider": "chroma"}
        vectorstore = get_vector_store(config)
        # Test that the factory returns a DefaultVectorStore
        assert hasattr(vectorstore, 'providers')
        # Test that it can get the right provider
        provider = vectorstore._get_provider(config)
        assert isinstance(provider, ChromaDBVectorStore)

    def test_get_default_vectorstore(self):
        """Test getting default vector store."""
        config = {}
        vectorstore = get_vector_store(config)
        
        # Should return DefaultVectorStore
        assert isinstance(vectorstore, DefaultVectorStore)
        
        # Check that it can get the default provider (faiss)
        provider = vectorstore._get_provider(config)
        assert isinstance(provider, FAISSVectorStore)

    def test_get_vectorstore_invalid_provider(self):
        """Test error handling for invalid provider."""
        config = {"provider": "nonexistent_provider"}
        vectorstore = get_vector_store(config)
        
        # The error should be raised when trying to get the provider
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            vectorstore._get_provider(config)
