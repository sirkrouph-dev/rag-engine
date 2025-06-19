# Vectorstore interface and implementations

import os
import time
import uuid
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC
import importlib
from rag_engine.core.base import BaseVectorStore

logger = logging.getLogger(__name__)

class VectorStoreProvider(ABC):
    """Base class for different vector store providers."""
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to the vector store."""
        raise NotImplementedError
        
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        raise NotImplementedError
        
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from the vector store by id."""
        raise NotImplementedError
        
    def persist(self, config: Dict[str, Any]) -> None:
        """Persist the vector store to disk if applicable."""
        raise NotImplementedError


class FAISSVectorStore(VectorStoreProvider):
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS package not installed. Run 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            raise
            
        self.index = None
        self.documents = []
        self.metadata = {}
    
    def _create_index(self, dimension: int, config: Dict[str, Any]) -> None:
        """Create a FAISS index with the specified configuration."""
        index_type = config.get("index_type", "Flat")
        metric = config.get("metric", "cosine").lower()
        
        if metric == "cosine":
            # Use inner product for cosine (vectors should be normalized)
            index = self.faiss.IndexFlatIP(dimension)
        elif metric == "l2" or metric == "euclidean":
            index = self.faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
        
        # Wrap with more efficient index types if specified
        if index_type == "IVF":
            # IVF requires training, better for larger datasets
            nlist = config.get("nlist", 100)  # Number of clusters
            quantizer = index
            index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist, 
                                          self.faiss.METRIC_INNER_PRODUCT if metric == "cosine" else self.faiss.METRIC_L2)
            # IVF indexes need training
            self.needs_training = True
        elif index_type == "HNSW":
            # Hierarchical NSW, good balance of speed and accuracy
            M = config.get("M", 16)  # Number of connections per layer
            index_hnsw = self.faiss.IndexHNSWFlat(dimension, M,
                                               self.faiss.METRIC_INNER_PRODUCT if metric == "cosine" else self.faiss.METRIC_L2)
            index = index_hnsw
            self.needs_training = False
        else:
            # Flat index doesn't need training
            self.needs_training = False
        
        self.index = index
    
    def _normalize_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        vectors_np = np.array(vectors).astype('float32')
        # Normalize vectors for cosine similarity
        faiss_normalize = getattr(self.faiss, "normalize_L2", None)
        if faiss_normalize:
            faiss_normalize(vectors_np)
        else:
            # Fallback normalization if faiss.normalize_L2 is not available
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            vectors_np = vectors_np / np.maximum(norms, 1e-10)
        return vectors_np
    
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to the FAISS index."""
        if not documents:
            logger.warning("No documents to add to FAISS index.")
            return
            
        # Extract embeddings
        embeddings = [doc.get("embedding", []) for doc in documents]
        if not embeddings or len(embeddings[0]) == 0:
            raise ValueError("Documents must have embeddings to be added to the vector store.")
        
        # Convert to numpy array
        dimension = len(embeddings[0])
        vectors = np.array(embeddings).astype('float32')
        
        # Create index if it doesn't exist
        if self.index is None:
            self._create_index(dimension, config)
            
        # Normalize vectors if using cosine similarity
        metric = config.get("metric", "cosine").lower()
        if metric == "cosine":
            vectors = self._normalize_vectors(embeddings)
        
        # Train the index if necessary (for IVF indexes)
        if self.needs_training and self.index.ntotal == 0:
            logger.info("Training FAISS index...")
            self.index.train(vectors)
        
        # Assign IDs to documents if not present
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc["id"] = str(uuid.uuid4())
        
        # Store document metadata
        doc_ids = [doc["id"] for doc in documents]
        for i, doc_id in enumerate(doc_ids):
            doc_without_embedding = {k: v for k, v in documents[i].items() if k != "embedding"}
            self.metadata[doc_id] = doc_without_embedding
        
        # Add vectors to the index
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # Map FAISS internal IDs to document IDs
        for i, doc_id in enumerate(doc_ids):
            self.documents.append(doc_id)
            
        logger.info(f"Added {len(documents)} documents to FAISS index. Total documents: {len(self.documents)}")
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the FAISS index for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty. No results to return.")
            return []
            
        top_k = config.get("top_k", 5)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize query vector for cosine similarity
        metric = config.get("metric", "cosine").lower()
        if metric == "cosine":
            query_vector = self._normalize_vectors([query_embedding])
        
        # For IVF indexes, set nprobe (number of clusters to visit)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = config.get("nprobe", 10)
        
        # Search the index
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no result found
                doc_id = self.documents[idx]
                doc_data = self.metadata.get(doc_id, {})
                
                # Convert distance to similarity score (for cosine or inner product)
                score = float(distances[0][i])
                if metric == "cosine" or metric == "inner":
                    # Convert distance to similarity [0, 1]
                    score = (score + 1) / 2
                elif metric == "l2" or metric == "euclidean":
                    # Convert L2 distance to similarity score (closer to 1 is better)
                    score = 1.0 / (1.0 + score)
                
                results.append({
                    "id": doc_id,
                    "score": score,
                    "document": doc_data
                })
                
        return results
    
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from the index by ID - not directly supported in FAISS."""
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        if not ids or self.index is None:
            return
            
        # Get all documents that should remain
        keep_docs = [doc_id for doc_id in self.documents if doc_id not in ids]
        keep_indices = [i for i, doc_id in enumerate(self.documents) if doc_id in keep_docs]
        
        if not keep_docs:
            # If no documents remain, reset the index
            self.index = None
            self.documents = []
            self.metadata = {}
            return
            
        # Get vectors to keep
        if hasattr(self.index, "reconstruct_batch"):
            # For indexes that support reconstruction
            keep_vectors = np.vstack([self.index.reconstruct(i) for i in keep_indices])
        else:
            # For indexes that don't support reconstruction, we need to rebuild from scratch
            logger.warning("FAISS index doesn't support reconstruction. Rebuilding index from scratch.")
            self.index = None
            
            # Rebuild with remaining documents
            remaining_docs = [self.metadata[doc_id] for doc_id in keep_docs if doc_id in self.metadata]
            
            # Remove deleted documents from metadata
            for doc_id in ids:
                self.metadata.pop(doc_id, None)
            
            # Reset documents list
            self.documents = []
            
            # Re-add remaining documents
            self.add(remaining_docs, config)
            return
            
        # Create a new index with the same parameters
        dimension = self.index.d
        new_index = None
        
        # Determine index type and recreate
        if isinstance(self.index, self.faiss.IndexFlat):
            metric = self.faiss.METRIC_L2
            if hasattr(self.index, 'metric_type'):
                metric = self.index.metric_type
            new_index = self.faiss.IndexFlat(dimension, metric)
        elif hasattr(self.faiss, 'IndexIVFFlat') and isinstance(self.index, self.faiss.IndexIVFFlat):
            quantizer = self.faiss.IndexFlat(dimension)
            metric = self.index.metric_type
            nlist = self.index.nlist
            new_index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist, metric)
            new_index.train(keep_vectors)
        elif hasattr(self.faiss, 'IndexHNSWFlat') and isinstance(self.index, self.faiss.IndexHNSWFlat):
            M = self.index.hnsw.M
            metric = self.index.metric_type
            new_index = self.faiss.IndexHNSWFlat(dimension, M, metric)
            
        if new_index is not None:
            # Add the kept vectors to the new index
            new_index.add(keep_vectors)
            self.index = new_index
            
            # Update the documents list and metadata
            self.documents = keep_docs
            for doc_id in ids:
                self.metadata.pop(doc_id, None)
                
            logger.info(f"Deleted {len(ids)} documents from FAISS index. Remaining documents: {len(self.documents)}")
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Persist the FAISS index and metadata to disk."""
        persist_dir = config.get("persist_directory")
        if not persist_dir:
            logger.warning("No persist directory specified. FAISS index will not be saved.")
            return
            
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Save the FAISS index
        index_path = persist_path / "faiss.index"
        if self.index is not None:
            self.faiss.write_index(self.index, str(index_path))
            
        # Save the metadata and document IDs
        metadata_path = persist_path / "faiss_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata,
                "documents": self.documents
            }, f)
            
        logger.info(f"FAISS index and metadata persisted to {persist_dir}")
    
    def load(self, config: Dict[str, Any]) -> None:
        """Load a persisted FAISS index and metadata from disk."""
        persist_dir = config.get("persist_directory")
        if not persist_dir:
            logger.warning("No persist directory specified. Cannot load FAISS index.")
            return
            
        persist_path = Path(persist_dir)
        index_path = persist_path / "faiss.index"
        metadata_path = persist_path / "faiss_metadata.pkl"
        
        # Check if files exist
        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"FAISS index or metadata file not found in {persist_dir}.")
            return
            
        # Load the FAISS index
        self.index = self.faiss.read_index(str(index_path))
        
        # Load the metadata and document IDs
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.documents = data["documents"]
            
        logger.info(f"Loaded FAISS index with {len(self.documents)} documents from {persist_dir}")


class ChromaDBVectorStore(VectorStoreProvider):
    """ChromaDB-based vector store with persistence and rich functionality."""
    
    def __init__(self):
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            logger.error("ChromaDB package not installed. Run 'pip install chromadb'")
            raise
            
        self.client = None
        self.collection = None
        
    def _get_client(self, config: Dict[str, Any]):
        """Get or create ChromaDB client based on config."""
        if self.client is not None:
            return self.client
        
        persist_dir = config.get("persist_directory")
        host = config.get("host")
        port = config.get("port")
        
        # Choose client type based on configuration
        if host and port:
            # Use HTTP client
            self.client = self.chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to ChromaDB server at {host}:{port}")
        elif persist_dir:
            # Use persistent client
            self.client = self.chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Created persistent ChromaDB client at {persist_dir}")
        else:
            # Use in-memory client
            self.client = self.chromadb.Client()
            logger.info("Created in-memory ChromaDB client")
            
        return self.client
    
    def _get_collection(self, config: Dict[str, Any]):
        """Get or create ChromaDB collection based on config."""
        if self.collection is not None:
            return self.collection
            
        client = self._get_client(config)
        collection_name = config.get("collection_name", "rag_engine_collection")
        
        # Get or create collection
        try:
            # Get existing collection
            self.collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing ChromaDB collection: {collection_name}")
        except Exception:
            # Create new collection
            # Get embedding dimensions from first document if not specified
            embedding_fn = None
            embedding_dimension = config.get("embedding_dimension", None)
              # For distance metrics
            metric = config.get("metric", "cosine")
            if metric == "cosine":
                distance_func = "cosine"
            elif metric in ["l2", "euclidean"]:
                distance_func = "l2"
            elif metric in ["inner", "dot"]:
                distance_func = "dot"
            else:
                distance_func = "cosine"
                
            self.collection = client.create_collection(
                name=collection_name,
                metadata={"description": "RAG Engine collection"}
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
            
        return self.collection
    
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to ChromaDB."""
        if not documents:
            logger.warning("No documents to add to ChromaDB.")
            return
            
        collection = self._get_collection(config)
        
        # Extract embeddings and document content
        ids = []
        embeddings = []
        metadatas = []
        contents = []
        
        for doc in documents:
            # Generate ID if not present
            if "id" not in doc:
                doc["id"] = str(uuid.uuid4())
            
            doc_id = doc["id"]
            doc_embedding = doc.get("embedding")
            
            if not doc_embedding:
                logger.warning(f"Document {doc_id} has no embedding. Skipping.")
                continue
                
            # Extract metadata and content
            metadata = {k: v for k, v in doc.items() 
                      if k not in ["id", "embedding", "content"] 
                      and not isinstance(v, (list, dict))}  # Exclude nested structures
            
            ids.append(doc_id)
            embeddings.append(doc_embedding)
            metadatas.append(metadata)
            contents.append(doc.get("content", ""))
        
        # Add documents in batches
        batch_size = config.get("batch_size", 100)
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_contents
            )
            
        logger.info(f"Added {len(ids)} documents to ChromaDB collection")
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the ChromaDB collection for similar documents."""
        collection = self._get_collection(config)
        
        # Query parameters
        top_k = config.get("top_k", 5)
        filters = config.get("filters")  # ChromaDB supports filtering
        include_metadata = config.get("include_metadata", True)
        include_documents = config.get("include_documents", True)
        
        # Query the collection
        query_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include_metadata=include_metadata,
            include_documents=include_documents
        )
        
        # Format results
        results = []
        if query_results and query_results["ids"]:
            for i, doc_id in enumerate(query_results["ids"][0]):
                score = float(query_results["distances"][0][i]) if "distances" in query_results else None
                
                if "distances" in query_results and score is not None:
                    # Convert distance to similarity score
                    metric = config.get("metric", "cosine")
                    if metric in ["cosine", "l2", "euclidean"]:
                        # For distance metrics (lower is better), convert to similarity [0, 1]
                        score = 1.0 / (1.0 + score)
                
                document = {}
                if include_documents:
                    document["content"] = query_results["documents"][0][i]
                if include_metadata:
                    document.update(query_results["metadatas"][0][i])
                
                results.append({
                    "id": doc_id,
                    "score": score,
                    "document": document
                })
                
        return results
    
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from ChromaDB by ID."""
        collection = self._get_collection(config)
        
        # Delete documents in batches
        batch_size = min(len(ids), 100)  # ChromaDB may have limits
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            collection.delete(ids=batch_ids)
            
        logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Persist the ChromaDB collection to disk if using persistent client."""
        if self.client is not None and hasattr(self.client, "persist"):
            self.client.persist()
            persist_dir = config.get("persist_directory", "Not specified")
            logger.info(f"ChromaDB collection persisted to {persist_dir}")
    
    def load(self, config: Dict[str, Any]) -> None:
        """Load ChromaDB collection (happens automatically when getting collection)."""
        self._get_collection(config)
        logger.info("ChromaDB collection loaded")


class PostgresVectorStore(VectorStoreProvider):
    """PostgreSQL-based vector store using pgvector extension."""
    
    def __init__(self):
        try:
            import psycopg2
            self.psycopg2 = psycopg2
        except ImportError:
            logger.error("psycopg2 package not installed. Run 'pip install psycopg2-binary'")
            raise
            
        self.conn = None
        self.table_name = None
    
    def _get_connection(self, config: Dict[str, Any]):
        """Get PostgreSQL connection."""
        if self.conn is not None and not self.conn.closed:
            return self.conn
            
        # Get connection parameters
        host = config.get("host", "localhost")
        port = config.get("port", 5432)
        database = config.get("database", "postgres")
        user = config.get("user", "postgres")
        password = config.get("password") or os.environ.get("PGPASSWORD")
        
        if not password:
            raise ValueError("PostgreSQL password must be provided in config or PGPASSWORD environment variable")
            
        # Connect to PostgreSQL
        self.conn = self.psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        logger.info(f"Connected to PostgreSQL at {host}:{port}/{database}")
        return self.conn
    
    def _setup_table(self, config: Dict[str, Any], embedding_dim: int):
        """Set up the vector table if it doesn't exist."""
        self.table_name = config.get("table_name", "rag_engine_vectors")
        
        conn = self._get_connection(config)
        with conn.cursor() as cursor:
            # Check if pgvector extension is installed
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if not cursor.fetchone():
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            
            # Create table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({embedding_dim}),
                    metadata JSONB,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index if it doesn't exist
            metric = config.get("metric", "cosine").lower()
            
            # Check if index exists
            cursor.execute(f"""
                SELECT 1 FROM pg_indexes 
                WHERE tablename = '{self.table_name.lower()}' 
                AND indexname = '{self.table_name.lower()}_vector_idx'
            """)
            
            if not cursor.fetchone():
                # Create appropriate index based on metric
                if metric == "cosine":
                    # For cosine similarity
                    cursor.execute(f"""
                        CREATE INDEX {self.table_name}_vector_idx 
                        ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
                elif metric in ["l2", "euclidean"]:
                    # For L2 distance
                    cursor.execute(f"""
                        CREATE INDEX {self.table_name}_vector_idx 
                        ON {self.table_name} USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100)
                    """)
                elif metric in ["inner", "dot"]:
                    # For inner product
                    cursor.execute(f"""
                        CREATE INDEX {self.table_name}_vector_idx 
                        ON {self.table_name} USING ivfflat (embedding vector_ip_ops)
                        WITH (lists = 100)
                    """)
                    
            conn.commit()
            logger.info(f"PostgreSQL table {self.table_name} set up with vector index")
    
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to PostgreSQL."""
        if not documents:
            logger.warning("No documents to add to PostgreSQL.")
            return
            
        # Ensure documents have embeddings
        for doc in documents:
            if "embedding" not in doc:
                raise ValueError("Documents must have embeddings to be added to the vector store.")
                
        # Get embedding dimension from first document
        embedding_dim = len(documents[0]["embedding"])
        
        # Set up table if needed
        self._setup_table(config, embedding_dim)
        
        # Prepare documents for insertion
        conn = self._get_connection(config)
        with conn.cursor() as cursor:
            for doc in documents:
                # Generate ID if not present
                if "id" not in doc:
                    doc["id"] = str(uuid.uuid4())
                    
                doc_id = doc["id"]
                embedding = doc["embedding"]
                content = doc.get("content", "")
                
                # Extract metadata
                metadata = {k: v for k, v in doc.items() 
                          if k not in ["id", "embedding", "content"]}
                
                # Insert document
                cursor.execute(f"""
                    INSERT INTO {self.table_name} (id, embedding, metadata, content)
                    VALUES (%s, %s::vector, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content,
                        created_at = CURRENT_TIMESTAMP
                """, (doc_id, embedding, metadata, content))
                
            conn.commit()
            logger.info(f"Added {len(documents)} documents to PostgreSQL vector store")
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the PostgreSQL vector store for similar documents."""
        if not self.table_name:
            logger.warning("Vector table not set up. Call add() first.")
            return []
            
        top_k = config.get("top_k", 5)
        metric = config.get("metric", "cosine").lower()
        
        conn = self._get_connection(config)
        with conn.cursor() as cursor:
            # Choose query operator based on metric
            operator = "<=>"  # Cosine distance operator
            order = "ASC"     # Lower distance is better (ascending)
            
            if metric == "cosine":
                operator = "<=>"  # Cosine distance
            elif metric in ["l2", "euclidean"]:
                operator = "<->"  # L2 distance
            elif metric in ["inner", "dot"]:
                operator = "<#>"  # Negative inner product
                order = "DESC"    # Higher is better for inner product (descending)
            
            # Apply filters if provided
            filter_clause = ""
            filter_params = []
            
            if "filters" in config:
                filters = config.get("filters", {})
                if filters:
                    conditions = []
                    
                    for key, value in filters.items():
                        conditions.append(f"metadata->>'%s' = %s")
                        filter_params.extend([key, str(value)])
                        
                    if conditions:
                        filter_clause = "WHERE " + " AND ".join(conditions)
            
            # Execute query
            query = f"""
                SELECT id, content, metadata, embedding {operator} %s::vector AS distance
                FROM {self.table_name}
                {filter_clause}
                ORDER BY distance {order}
                LIMIT %s
            """
            
            cursor.execute(query, [query_embedding, *filter_params, top_k])
            results = []
            
            for row in cursor.fetchall():
                doc_id, content, metadata_json, distance = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Convert distance to similarity score [0, 1]
                score = None
                if metric == "cosine":
                    score = 1.0 - distance  # Cosine distance to similarity
                elif metric in ["l2", "euclidean"]:
                    score = 1.0 / (1.0 + distance)  # L2 distance to similarity
                elif metric in ["inner", "dot"]:
                    # Assuming normalized vectors for inner product
                    # Shift from [-1, 1] to [0, 1]
                    score = (distance + 1.0) / 2.0
                
                document = {"content": content}
                document.update(metadata)
                
                results.append({
                    "id": doc_id,
                    "score": score,
                    "document": document
                })
                
            return results
    
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from PostgreSQL by ID."""
        if not self.table_name:
            logger.warning("Vector table not set up. Nothing to delete.")
            return
            
        conn = self._get_connection(config)
        with conn.cursor() as cursor:
            # PostgreSQL allows parameterized IN clauses
            placeholders = ", ".join(["%s"] * len(ids))
            cursor.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id IN ({placeholders})
            """, ids)
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Deleted {deleted_count} documents from PostgreSQL vector store")
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Persist is automatic in PostgreSQL."""
        logger.info("PostgreSQL vector store is automatically persisted")
    
    def load(self, config: Dict[str, Any]) -> None:
        """Loading happens automatically when querying."""
        self.table_name = config.get("table_name", "rag_engine_vectors")
        logger.info(f"PostgreSQL vector store table set to {self.table_name}")


class PineconeVectorStore(VectorStoreProvider):
    """Pinecone vector store for cloud-based vector search."""
    
    def __init__(self):
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            logger.error("Pinecone package not installed. Run 'pip install pinecone-client'")
            raise
            
        self.index = None
    
    def _initialize_pinecone(self, config: Dict[str, Any]):
        """Initialize Pinecone client."""
        api_key = config.get("api_key") or os.environ.get("PINECONE_API_KEY")
        environment = config.get("environment") or os.environ.get("PINECONE_ENVIRONMENT")
        
        if not api_key:
            raise ValueError("Pinecone API key must be provided in config or PINECONE_API_KEY environment variable")
        if not environment:
            raise ValueError("Pinecone environment must be provided in config or PINECONE_ENVIRONMENT environment variable")
            
        self.pinecone.init(api_key=api_key, environment=environment)
    
    def _get_index(self, config: Dict[str, Any]):
        """Get or create Pinecone index."""
        if self.index is not None:
            return self.index
            
        self._initialize_pinecone(config)
        
        index_name = config.get("index_name", "rag-engine-index")
        dimension = config.get("dimension")
        metric = config.get("metric", "cosine")
        
        # Convert metric name if needed
        if metric == "l2" or metric == "euclidean":
            metric = "euclidean"
        elif metric == "inner" or metric == "dot":
            metric = "dotproduct"
        else:
            metric = "cosine"
        
        # Check if index exists
        if index_name in self.pinecone.list_indexes():
            self.index = self.pinecone.Index(index_name)
            logger.info(f"Connected to existing Pinecone index: {index_name}")
        else:
            # Create index if dimension is provided
            if not dimension:
                raise ValueError("Dimension must be provided when creating a new Pinecone index")
                
            self.pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                metadata_config={"indexed": config.get("metadata_fields", [])}
            )
            
            # Wait for index to be ready
            while not index_name in self.pinecone.list_indexes():
                logger.info("Waiting for Pinecone index to be ready...")
                time.sleep(5)
                
            self.index = self.pinecone.Index(index_name)
            logger.info(f"Created new Pinecone index: {index_name}")
            
        return self.index
    
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to Pinecone."""
        if not documents:
            logger.warning("No documents to add to Pinecone.")
            return
            
        # Set dimension from first document if not provided in config
        if "dimension" not in config and documents:
            config["dimension"] = len(documents[0].get("embedding", []))
            
        index = self._get_index(config)
        
        # Prepare vectors for upsert
        vectors = []
        for doc in documents:
            # Generate ID if not present
            if "id" not in doc:
                doc["id"] = str(uuid.uuid4())
                
            doc_id = doc["id"]
            embedding = doc.get("embedding", [])
            
            # Prepare metadata (exclude large fields)
            metadata = {k: v for k, v in doc.items() 
                      if k not in ["id", "embedding"] 
                      and not isinstance(v, (list, dict))
                      and len(str(v)) < 40000}  # Pinecone has size limits
                      
            # Add content to metadata if present
            if "content" in doc:
                metadata["content"] = doc["content"]
                
            vectors.append((doc_id, embedding, metadata))
            
        # Upsert in batches
        batch_size = 100  # Pinecone batch size limit
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=[(id, emb, meta) for id, emb, meta in batch])
            
        logger.info(f"Added {len(documents)} documents to Pinecone index")
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the Pinecone index for similar documents."""
        index = self._get_index(config)
        
        top_k = config.get("top_k", 5)
        namespace = config.get("namespace", "")
        include_metadata = config.get("include_metadata", True)
        filter = config.get("filters")  # Pinecone filter syntax
        
        # Query the index
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=namespace,
            filter=filter
        )
        
        # Format results
        results = []
        for match in query_results.matches:
            document = {}
            if include_metadata and hasattr(match, "metadata"):
                document.update(match.metadata)
                
            results.append({
                "id": match.id,
                "score": match.score,
                "document": document
            })
            
        return results
    
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from Pinecone by ID."""
        index = self._get_index(config)
        namespace = config.get("namespace", "")
        
        # Delete in batches
        batch_size = 1000  # Pinecone limit for delete operations
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            index.delete(ids=batch_ids, namespace=namespace)
            
        logger.info(f"Deleted {len(ids)} documents from Pinecone index")
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Pinecone data is automatically persisted."""
        logger.info("Pinecone index is automatically persisted")
    
    def load(self, config: Dict[str, Any]) -> None:
        """Loading happens automatically when connecting to index."""
        self._get_index(config)
        logger.info("Pinecone index loaded")


class QdrantVectorStore(VectorStoreProvider):
    """Qdrant-based vector store for efficient similarity search."""
    
    def __init__(self):
        try:
            import qdrant_client
            from qdrant_client.http import models as rest
            self.qdrant_client = qdrant_client
            self.rest = rest
        except ImportError:
            logger.error("Qdrant client package not installed. Run 'pip install qdrant-client'")
            raise
            
        self.client = None
        self.collection_name = None
        
    def _get_client(self, config: Dict[str, Any]):
        """Get or initialize Qdrant client."""
        if self.client is not None:
            return self.client
            
        # Determine connection method
        url = config.get("url")
        host = config.get("host", "localhost")
        port = config.get("port", 6333)
        api_key = config.get("api_key") or os.environ.get("QDRANT_API_KEY")
        prefer_grpc = config.get("prefer_grpc", False)
        timeout = config.get("timeout", 30)
        
        if url:
            # Connect to cloud instance with URL
            self.client = self.qdrant_client.QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )
            logger.info(f"Connected to Qdrant at {url}")
        else:
            # Connect to local instance or specified host/port
            self.client = self.qdrant_client.QdrantClient(
                host=host,
                port=port,
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
        return self.client
        
    def _setup_collection(self, config: Dict[str, Any], vector_size: int):
        """Create collection if it doesn't exist."""
        self.collection_name = config.get("collection_name", "rag_engine_collection")
        client = self._get_client(config)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name in collection_names:
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            return
            
        # Set up distance metric
        metric = config.get("metric", "cosine").lower()
        if metric == "cosine":
            distance = self.rest.Distance.COSINE
        elif metric in ["l2", "euclidean"]:
            distance = self.rest.Distance.EUCLID
        elif metric in ["inner", "dot"]:
            distance = self.rest.Distance.DOT
        else:
            distance = self.rest.Distance.COSINE
            
        # Create collection
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.rest.VectorParams(
                size=vector_size,
                distance=distance
            ),
            optimizers_config=self.rest.OptimizersConfigDiff(
                indexing_threshold=config.get("indexing_threshold", 10000)
            )
        )
        
        logger.info(f"Created new Qdrant collection: {self.collection_name}")
        
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents with embeddings to Qdrant."""
        if not documents:
            logger.warning("No documents to add to Qdrant.")
            return
            
        # Extract embedding dimension from first document
        if "embedding" not in documents[0]:
            raise ValueError("Documents must have embeddings to be added to Qdrant")
            
        vector_size = len(documents[0]["embedding"])
        
        # Setup collection
        self._setup_collection(config, vector_size)
        
        # Prepare points for upsert
        client = self._get_client(config)
        points = []
        
        for doc in documents:
            if "id" not in doc:
                doc["id"] = str(uuid.uuid4())
                
            # Extract and handle point ID
            point_id = doc["id"]
            if isinstance(point_id, str) and not point_id.isdigit():
                # Qdrant supports both UUID strings and integers for IDs
                point_id_obj = point_id
            else:
                # Convert to int if it's a digit string or already an int
                point_id_obj = int(point_id)
                
            # Extract embedding
            embedding = doc["embedding"]
            
            # Extract payload (metadata)
            payload = {k: v for k, v in doc.items() if k not in ["id", "embedding"]}
            
            points.append(self.rest.PointStruct(
                id=point_id_obj,
                vector=embedding,
                payload=payload
            ))
            
        # Upsert in batches
        batch_size = config.get("batch_size", 100)
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            
        logger.info(f"Added {len(documents)} documents to Qdrant collection")
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query Qdrant for similar documents."""
        top_k = config.get("top_k", 5)
        client = self._get_client(config)
        
        # Get filter condition if provided
        filter_condition = None
        if "filters" in config and config["filters"]:
            filter_condition = self._build_filter(config["filters"])
            
        search_result = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_condition
        )
        
        # Format results
        results = []
        for scored_point in search_result:
            # Get document data
            payload = scored_point.payload or {}
            
            results.append({
                "id": str(scored_point.id),
                "score": scored_point.score,
                "document": payload
            })
            
        return results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Any]:
        """Build a Qdrant filter from provided filter dict."""
        if not filters:
            return None
            
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle list values - create an OR condition
                conditions.append(
                    self.rest.FieldCondition(
                        key=key,
                        match=self.rest.MatchAny(any=value)
                    )
                )
            else:
                # Handle single values - exact match
                conditions.append(
                    self.rest.FieldCondition(
                        key=key,
                        match=self.rest.MatchValue(value=value)
                    )
                )
                
        # Combine all conditions with AND
        if conditions:
            return self.rest.Filter(
                must=conditions
            )
        return None
        
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from Qdrant by ID."""
        if not ids:
            return
            
        client = self._get_client(config)
        
        # Convert string IDs to appropriate format
        formatted_ids = []
        for id_val in ids:
            if isinstance(id_val, str) and not id_val.isdigit():
                # Keep as string for UUID
                formatted_ids.append(id_val)
            else:
                # Convert to int
                formatted_ids.append(int(id_val))
                
        # Delete in batches
        batch_size = config.get("batch_size", 100)
        for i in range(0, len(formatted_ids), batch_size):
            batch_ids = formatted_ids[i:i+batch_size]
            client.delete(
                collection_name=self.collection_name,
                points_selector=self.rest.PointIdsList(
                    points=batch_ids
                )
            )
            
        logger.info(f"Deleted {len(ids)} documents from Qdrant collection")
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Persistence is automatic in Qdrant."""
        logger.info("Qdrant collection is automatically persisted")
    
    def load(self, config: Dict[str, Any]) -> None:
        """Loading happens automatically when connecting."""
        self.collection_name = config.get("collection_name", "rag_engine_collection")
        self._get_client(config)
        logger.info(f"Qdrant collection set to {self.collection_name}")


class DefaultVectorStore(BaseVectorStore):
    """Main vector store class that delegates to the appropriate provider based on config."""
    
    def __init__(self):
        # Use lazy loading for providers to avoid import errors
        self.providers = {}
        self.available_providers = {
            "faiss": FAISSVectorStore,
            "chroma": ChromaDBVectorStore,
            "postgres": PostgresVectorStore,
            "pinecone": PineconeVectorStore,
            "qdrant": QdrantVectorStore
        }
        
    def _get_provider(self, config: Dict[str, Any]) -> VectorStoreProvider:
        """Get the configured vector store provider with lazy loading."""
        provider_name = config.get("provider", "faiss").lower()
        
        if provider_name not in self.available_providers:
            raise ValueError(f"Unsupported vector store provider: {provider_name}")
        
        # Lazy load the provider
        if provider_name not in self.providers:
            try:
                provider_class = self.available_providers[provider_name]
                self.providers[provider_name] = provider_class()
            except Exception as e:
                raise ValueError(f"Failed to initialize {provider_name} provider: {str(e)}")
            
        return self.providers[provider_name]
    
    def add(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """Add documents to the vector store."""
        provider = self._get_provider(config)
        provider.add(documents, config)
    
    def query(self, query_embedding: List[float], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        provider = self._get_provider(config)
        return provider.query(query_embedding, config)
    
    def delete(self, ids: List[str], config: Dict[str, Any]) -> None:
        """Delete documents from the vector store by ID."""
        provider = self._get_provider(config)
        provider.delete(ids, config)
    
    def persist(self, config: Dict[str, Any]) -> None:
        """Persist the vector store if supported."""
        provider = self._get_provider(config)
        provider.persist(config)
        
    def load(self, config: Dict[str, Any]) -> None:
        """Load a persisted vector store if supported."""
        provider = self._get_provider(config)
        provider.load(config)


# Factory function to get the appropriate vector store
def get_vector_store(config: Dict[str, Any]) -> BaseVectorStore:
    """Get a vector store based on configuration."""
    vector_store = DefaultVectorStore()
    
    # Load existing data if available
    if config.get("persist_directory") or config.get("host"):
        try:
            vector_store.load(config)
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
    
    return vector_store
