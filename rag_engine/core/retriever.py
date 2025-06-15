# Retriever interface and implementations

import logging
import heapq
import numpy as np
from abc import ABC
from typing import List, Dict, Any, Optional, Union, Callable
from rag_engine.core.base import BaseRetriever
from rag_engine.core.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class RetrieverStrategy(ABC):
    """Base class for different retrieval strategies."""
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using the strategy."""
        raise NotImplementedError


class SimpleRetriever(RetrieverStrategy):
    """Simple vector similarity search retriever."""
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents based on vector similarity."""
        top_k = config.get("top_k", 5)
        
        # Set up query config for the vector store
        query_config = {
            "top_k": top_k
        }
        
        # Add any vector store specific filters from the config
        if "filters" in config:
            query_config["filters"] = config["filters"]
            
        # Perform the similarity search
        results = vectorstore.query(query_embedding, query_config)
        logger.info(f"Retrieved {len(results)} documents with simple similarity search")
        
        return results


class ThresholdRetriever(RetrieverStrategy):
    """Similarity search with a minimum similarity threshold."""
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents with similarity above the threshold."""
        top_k = config.get("top_k", 20)  # Start with more results to filter
        threshold = config.get("similarity_threshold", 0.7)
        
        # Set up query config for the vector store
        query_config = {
            "top_k": top_k
        }
        
        # Add any vector store specific filters
        if "filters" in config:
            query_config["filters"] = config["filters"]
            
        # Perform the similarity search
        results = vectorstore.query(query_embedding, query_config)
        
        # Filter by similarity threshold
        filtered_results = [doc for doc in results if doc.get("score", 0) >= threshold]
        
        logger.info(f"Retrieved {len(filtered_results)} documents after threshold filtering (threshold={threshold})")
        return filtered_results


class MMRRetriever(RetrieverStrategy):
    """Maximum Marginal Relevance retriever for balancing relevance and diversity."""
    
    def _calculate_mmr(self, query_embedding: List[float], doc_embeddings: List[List[float]], 
                     already_selected_indices: List[int], lambda_param: float, k: int) -> int:
        """
        Calculate Maximum Marginal Relevance.
        
        Args:
            query_embedding: Embedding of the query
            doc_embeddings: List of document embeddings
            already_selected_indices: Indices of already selected documents
            lambda_param: Weight parameter for MMR calculation (0-1)
            k: Number of documents to select
            
        Returns:
            Index of the document with highest MMR score
        """
        if not already_selected_indices:
            # If nothing has been selected, select the most similar to the query
            similarities = [self._cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
            return similarities.index(max(similarities))
            
        # Calculate relevance scores (similarity to query)
        relevance_scores = [self._cosine_similarity(query_embedding, doc_embeddings[i]) 
                          for i in range(len(doc_embeddings)) if i not in already_selected_indices]
        
        # Calculate diversity scores (negative of maximum similarity to any already selected document)
        diversity_scores = []
        remaining_indices = [i for i in range(len(doc_embeddings)) if i not in already_selected_indices]
        
        for i in remaining_indices:
            similarity_to_selected = [self._cosine_similarity(doc_embeddings[i], doc_embeddings[j]) 
                                    for j in already_selected_indices]
            # Negative of maximum similarity (higher value means more diverse)
            diversity_score = -max(similarity_to_selected) if similarity_to_selected else 0
            diversity_scores.append(diversity_score)
            
        # Calculate MMR scores
        mmr_scores = [lambda_param * relevance_scores[idx] + (1 - lambda_param) * diversity_scores[idx] 
                    for idx in range(len(remaining_indices))]
        
        # Return the index of the document with the highest MMR score
        max_mmr_index = mmr_scores.index(max(mmr_scores))
        return remaining_indices[max_mmr_index]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1_np, vec2_np) / (norm1 * norm2)
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using Maximum Marginal Relevance."""
        top_k = config.get("top_k", 5)
        fetch_k = config.get("fetch_k", top_k * 3)  # Fetch more documents initially
        lambda_param = config.get("lambda_param", 0.7)  # Balance between relevance and diversity
        
        # Set up query config for the vector store
        query_config = {
            "top_k": fetch_k
        }
        
        # Add any vector store specific filters
        if "filters" in config:
            query_config["filters"] = config["filters"]
            
        # Perform the initial similarity search
        initial_results = vectorstore.query(query_embedding, query_config)
        
        if not initial_results:
            return []
            
        # Extract document embeddings
        doc_embeddings = []
        
        # Check if we have embeddings in the results
        if "embedding" in initial_results[0].get("document", {}):
            # If embeddings are already in the results
            doc_embeddings = [doc["document"]["embedding"] for doc in initial_results]
        else:
            # Need to reconstruct embeddings from the vector store if available
            if hasattr(vectorstore, "reconstruct_embeddings"):
                doc_ids = [doc["id"] for doc in initial_results]
                doc_embeddings = vectorstore.reconstruct_embeddings(doc_ids)
            else:
                # Fall back to simple retriever if no embeddings available for MMR
                logger.warning("MMR retrieval requires document embeddings, falling back to simple retrieval")
                return initial_results[:top_k]
        
        # Apply MMR to select diverse but relevant documents
        selected_indices = []
        for _ in range(min(top_k, len(initial_results))):
            next_idx = self._calculate_mmr(
                query_embedding, doc_embeddings, selected_indices, lambda_param, top_k
            )
            selected_indices.append(next_idx)
            
        # Return selected documents in order
        mmr_results = [initial_results[idx] for idx in selected_indices]
        logger.info(f"Retrieved {len(mmr_results)} documents using MMR")
        
        return mmr_results


class HybridRetriever(RetrieverStrategy):
    """Hybrid retrieval combining dense and sparse (keyword) search."""
    
    def _keyword_search(self, vectorstore: Any, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform keyword-based search if supported by the vector store."""
        # Check if vector store supports keyword search
        if hasattr(vectorstore, "keyword_search"):
            top_k = config.get("top_k", 5)
            return vectorstore.keyword_search(query, {"top_k": top_k})
        
        # Use full-text search in PostgreSQL if available
        elif hasattr(vectorstore, "text_search") and hasattr(vectorstore, "_get_connection"):
            top_k = config.get("top_k", 5)
            return vectorstore.text_search(query, {"top_k": top_k})
            
        # Fall back to empty results if keyword search not supported
        logger.warning("Keyword search not supported by the vector store")
        return []
    
    def _merge_results(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], 
                      alpha: float, top_k: int) -> List[Dict[str, Any]]:
        """Merge dense and sparse search results with weighting."""
        # Create a map of document IDs to their details
        merged_map = {}
        
        # Process dense results first
        for doc in dense_results:
            doc_id = doc["id"]
            merged_map[doc_id] = {
                "id": doc_id,
                "document": doc["document"],
                "dense_score": doc.get("score", 0),
                "sparse_score": 0,
                "combined_score": alpha * doc.get("score", 0)
            }
            
        # Combine with sparse results
        for doc in sparse_results:
            doc_id = doc["id"]
            if doc_id in merged_map:
                # Document exists in dense results
                merged_map[doc_id]["sparse_score"] = doc.get("score", 0)
                merged_map[doc_id]["combined_score"] += (1 - alpha) * doc.get("score", 0)
            else:
                # New document from sparse results
                merged_map[doc_id] = {
                    "id": doc_id,
                    "document": doc["document"],
                    "dense_score": 0,
                    "sparse_score": doc.get("score", 0),
                    "combined_score": (1 - alpha) * doc.get("score", 0)
                }
                
        # Sort by combined score and return top_k
        sorted_results = sorted(
            merged_map.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )[:top_k]
        
        # Format the results to match the expected output format
        final_results = []
        for doc in sorted_results:
            final_results.append({
                "id": doc["id"],
                "score": doc["combined_score"],
                "document": doc["document"]
            })
            
        return final_results
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid search (dense + sparse)."""
        top_k = config.get("top_k", 5)
        alpha = config.get("alpha", 0.5)  # Weight for dense vs sparse (0-1)
        
        # Set up query config for vector search
        query_config = {
            "top_k": top_k
        }
        
        # Add any vector store specific filters
        if "filters" in config:
            query_config["filters"] = config["filters"]
            
        # Perform dense vector search
        dense_results = vectorstore.query(query_embedding, query_config)
        
        # Perform sparse/keyword search
        sparse_results = self._keyword_search(vectorstore, query, {"top_k": top_k})
        
        # If sparse search failed or returned no results, just return dense results
        if not sparse_results:
            return dense_results
            
        # Merge and rerank results
        hybrid_results = self._merge_results(dense_results, sparse_results, alpha, top_k)
        
        logger.info(f"Retrieved {len(hybrid_results)} documents using hybrid search")
        return hybrid_results


class EnsembleRetriever(RetrieverStrategy):
    """Ensemble retriever that combines results from multiple strategies."""
    
    def __init__(self, strategies: List[RetrieverStrategy] = None):
        """Initialize with a list of retrieval strategies."""
        self.strategies = strategies or [
            SimpleRetriever(),
            MMRRetriever()
        ]
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using multiple strategies and combine results."""
        top_k = config.get("top_k", 5)
        weights = config.get("strategy_weights", [1.0] * len(self.strategies))
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Get results from each strategy
        all_results = []
        for i, strategy in enumerate(self.strategies):
            strategy_results = strategy.retrieve(vectorstore, query_embedding, query, config)
            
            # Apply strategy weight to scores
            for result in strategy_results:
                result["score"] = result.get("score", 0) * weights[i]
                
            all_results.extend(strategy_results)
            
        # Combine and deduplicate results
        merged_map = {}
        for result in all_results:
            doc_id = result["id"]
            if doc_id in merged_map:
                # Sum scores if document appears in multiple strategies
                merged_map[doc_id]["score"] += result["score"]
            else:
                merged_map[doc_id] = result
                
        # Sort by combined score and return top_k
        sorted_results = sorted(
            merged_map.values(), 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )[:top_k]
        
        logger.info(f"Retrieved {len(sorted_results)} documents using ensemble retrieval")
        return sorted_results


class RerankerRetriever(RetrieverStrategy):
    """Retriever that uses a separate model to rerank initial results."""
    
    def __init__(self, base_strategy: RetrieverStrategy = None, reranker_fn: Callable = None):
        """
        Initialize with base strategy and reranker function.
        
        Args:
            base_strategy: Base retrieval strategy to get initial candidates
            reranker_fn: Function to rerank results, signature: (query, docs) -> ranked_docs
        """
        self.base_strategy = base_strategy or SimpleRetriever()
        self.reranker_fn = reranker_fn
    
    def _default_reranker(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Default reranking method using BM25-style scoring."""
        # Simple keyword-based reranking if no custom reranker is provided
        query_terms = query.lower().split()
        
        for doc in documents:
            content = str(doc.get("document", {}).get("content", "")).lower()
            
            # Calculate a basic TF score
            score = 0
            for term in query_terms:
                if term in content:
                    score += content.count(term) / len(content.split())
            
            # Combine with original similarity score
            original_score = doc.get("score", 0)
            doc["score"] = (original_score + score) / 2
            
        # Sort by new score
        return sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using initial strategy then rerank."""
        top_k = config.get("top_k", 5)
        fetch_k = config.get("fetch_k", top_k * 3)  # Fetch more to rerank
        
        # Modify config for base strategy to fetch more candidates
        base_config = dict(config)
        base_config["top_k"] = fetch_k
        
        # Fetch initial candidates
        initial_results = self.base_strategy.retrieve(vectorstore, query_embedding, query, base_config)
        
        # Apply reranking
        if self.reranker_fn:
            reranked_results = self.reranker_fn(query, initial_results)
        else:
            reranked_results = self._default_reranker(query, initial_results)
            
        # Return top_k results after reranking
        return reranked_results[:top_k]


class SelfQueryRetriever(RetrieverStrategy):
    """Retriever that uses an LLM to generate filters from the query."""
    
    def __init__(self, llm_provider=None):
        """Initialize with optional LLM provider for generating filters."""
        self.llm_provider = llm_provider
        
    def _extract_filters(self, query: str, metadata_schema: Dict[str, Any], llm_provider: Any) -> Dict[str, Any]:
        """Extract filters from the query using an LLM."""
        if not llm_provider:
            # Simple keyword-based filter extraction as fallback
            return self._simple_filter_extraction(query, metadata_schema)
            
        # Generate a system prompt for filter extraction
        system_prompt = f"""
        Given the user query and the available metadata fields with their types, extract relevant filters.
        Return a JSON object with field names as keys and filter values as values.
        Available fields: {metadata_schema}
        """
        
        # Generate filters using LLM
        response = llm_provider.generate(
            prompt={
                "system": system_prompt,
                "user": f"Query: {query}"
            },
            config={"temperature": 0.1}  # Low temperature for consistent results
        )
        
        try:
            import json
            filters = json.loads(response)
            return filters
        except Exception as e:
            logger.error(f"Failed to parse LLM filter response: {e}")
            return {}
    
    def _simple_filter_extraction(self, query: str, metadata_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based filter extraction without using an LLM."""
        filters = {}
        
        # Look for common patterns like field:value
        query_lower = query.lower()
        
        for field, field_type in metadata_schema.items():
            field_lower = field.lower()
            
            # Check for field:value pattern
            pattern = f"{field_lower}:"
            if pattern in query_lower:
                start_idx = query_lower.index(pattern) + len(pattern)
                end_idx = query_lower.find(" ", start_idx)
                if end_idx == -1:  # If at the end of the query
                    end_idx = len(query_lower)
                    
                value = query[start_idx:end_idx].strip()
                
                # Convert value to appropriate type
                if field_type == "number":
                    try:
                        value = float(value)
                    except:
                        continue
                elif field_type == "boolean":
                    value = value.lower() in ["true", "yes", "1"]
                    
                filters[field] = value
                
        return filters
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents with filters extracted from the query."""
        top_k = config.get("top_k", 5)
        metadata_schema = config.get("metadata_schema", {})
        
        # Extract filters from query if metadata schema is provided
        filters = {}
        if metadata_schema:
            filters = self._extract_filters(query, metadata_schema, self.llm_provider)
            
        # Combine with existing filters
        if "filters" in config:
            filters.update(config["filters"])
            
        # Set up query config
        query_config = {
            "top_k": top_k,
            "filters": filters
        }
        
        # Perform the search
        results = vectorstore.query(query_embedding, query_config)
        logger.info(f"Retrieved {len(results)} documents with self-query filters: {filters}")
        
        return results


class DedupRetriever(RetrieverStrategy):
    """Retriever that removes duplicate/redundant content."""
    
    def __init__(self, base_strategy: RetrieverStrategy = None):
        """Initialize with base strategy."""
        self.base_strategy = base_strategy or SimpleRetriever()
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity for deduplication."""
        # Simple Jaccard similarity as a lightweight approach
        if not text1 or not text2:
            return 0
            
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def _remove_duplicates(self, results: List[Dict[str, Any]], similarity_threshold: float) -> List[Dict[str, Any]]:
        """Remove redundant documents based on content similarity."""
        if not results:
            return []
            
        # Sort by score to prioritize higher-scored documents
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        deduplicated = [sorted_results[0]]  # Keep the first document
        
        for doc in sorted_results[1:]:
            # Get document content
            content = str(doc.get("document", {}).get("content", ""))
            is_duplicate = False
            
            # Compare with all documents already in deduplicated list
            for existing_doc in deduplicated:
                existing_content = str(existing_doc.get("document", {}).get("content", ""))
                similarity = self._compute_similarity(content, existing_content)
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                deduplicated.append(doc)
                
        return deduplicated
    
    def retrieve(self, vectorstore: Any, query_embedding: List[float], query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents and remove duplicates."""
        top_k = config.get("top_k", 5)
        similarity_threshold = config.get("similarity_threshold", 0.85)
        fetch_k = config.get("fetch_k", top_k * 2)  # Fetch more to handle deduplication
        
        # Modify config for base strategy to fetch more candidates
        base_config = dict(config)
        base_config["top_k"] = fetch_k
        
        # Fetch initial results
        initial_results = self.base_strategy.retrieve(vectorstore, query_embedding, query, base_config)
        
        # Apply deduplication
        deduplicated_results = self._remove_duplicates(initial_results, similarity_threshold)
        
        # Return top_k results or fewer if deduplication removed several
        final_results = deduplicated_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} documents after deduplication")
        return final_results


class DefaultRetriever(BaseRetriever):
    """Main retriever that delegates to a specific strategy based on config."""
    
    def __init__(self):
        """Initialize with available retrieval strategies."""
        self.strategies = {
            "simple": SimpleRetriever(),
            "threshold": ThresholdRetriever(),
            "mmr": MMRRetriever(),
            "hybrid": HybridRetriever(),
            "ensemble": EnsembleRetriever(),
            "rerank": RerankerRetriever(),
            "self_query": SelfQueryRetriever(),
            "dedup": DedupRetriever()
        }
        
        # Default strategy
        self.default_strategy = "simple"
    
    def retrieve(self, vectorstore: Any, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using the configured strategy."""
        # Determine the retrieval strategy
        strategy_name = config.get("retrieval_strategy", self.default_strategy)
        
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown retrieval strategy: {strategy_name}, using {self.default_strategy}")
            strategy_name = self.default_strategy
            
        strategy = self.strategies[strategy_name]
        
        # Get embedder from config to embed the query
        embedder = config.get("embedder")
        if not embedder:
            raise ValueError("Embedder not provided in retriever config")
            
        # Embed the query
        query_embedding = embedder.embed_query(query, config.get("embedding_config", {}))
        
        # Perform retrieval with the selected strategy
        results = strategy.retrieve(vectorstore, query_embedding, query, config)
        
        # Format results if needed
        return results


# Factory function to get the appropriate retriever
def get_retriever(config: Dict[str, Any]) -> BaseRetriever:
    """Get a retriever based on configuration."""
    return DefaultRetriever()
