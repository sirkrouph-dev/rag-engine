"""
Alternative orchestrator implementations demonstrating modularity.

These orchestrators show how different RAG strategies can be implemented
without changing the API layer.
"""

from typing import Dict, Any, List
from rag_engine.core.orchestration import BaseOrchestrator, OrchestratorFactory
from rag_engine.config.schema import RAGConfig
from rag_engine.core.orchestration import ComponentRegistry


class HybridRetrievalOrchestrator(BaseOrchestrator):
    """Orchestrator that combines multiple retrieval methods."""
    
    def build(self) -> None:
        """Build pipeline with hybrid retrieval (semantic + BM25)."""
        print("🚀 Building RAG pipeline with HybridRetrievalOrchestrator...")
        
        try:
            # Create base components
            self._create_loader()
            self._create_chunker()
            self._create_embedder()
            self._create_vectorstore()
            
            # Create multiple retrievers
            self._create_semantic_retriever()
            self._create_bm25_retriever()
            
            self._create_llm()
            self._create_prompter()
            
            # Load and process documents
            self._load_documents()
            self._chunk_documents()
            self._embed_and_store()
            self._build_bm25_index()
            
            self._is_built = True
            print("✅ Hybrid retrieval pipeline built successfully")
            
        except Exception as e:
            print(f"❌ Pipeline build failed: {e}")
            raise
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute query using hybrid retrieval."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        
        try:
            # Get results from both retrievers
            semantic_retriever = self.get_component('semantic_retriever')
            bm25_retriever = self.get_component('bm25_retriever')
            
            semantic_docs = semantic_retriever.retrieve(query, top_k=5)
            bm25_docs = bm25_retriever.retrieve(query, top_k=5)
            
            # Combine and re-rank results
            combined_docs = self._combine_results(semantic_docs, bm25_docs)
            
            # Generate prompt and response
            prompter = self.get_component('prompter')
            prompt = prompter.generate_prompt(query, combined_docs)
            
            llm = self.get_component('llm')
            response = llm.generate(prompt)
            
            return {
                "answer": response,
                "sources": [doc.get("metadata", {}) for doc in combined_docs],
                "retrieval_method": "hybrid",
                "semantic_results": len(semantic_docs),
                "bm25_results": len(bm25_docs),
                "combined_results": len(combined_docs),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "orchestrator": "HybridRetrievalOrchestrator",
            "is_built": self._is_built,
            "components": list(self.components.keys()),
            "retrieval_methods": ["semantic", "bm25"],
            "config": {
                "embedding_provider": getattr(self.config.embedding, 'provider', 'default'),
                "vectorstore_provider": getattr(self.config.vectorstore, 'provider', 'default'),
                "llm_provider": getattr(self.config.llm, 'provider', 'default')
            }
        }
    
    def _create_semantic_retriever(self) -> None:
        """Create semantic similarity retriever."""
        vectorstore = self.get_component('vectorstore')
        retriever_config = {
            'method': 'similarity',
            'vectorstore': vectorstore,
            'top_k': self.config.retrieval.top_k
        }
        
        retriever = self.registry.create_component('retriever', 'similarity', retriever_config)
        self.add_component('semantic_retriever', retriever)
    
    def _create_bm25_retriever(self) -> None:
        """Create BM25 keyword retriever."""
        retriever_config = {
            'method': 'bm25',
            'top_k': self.config.retrieval.top_k
        }
        
        retriever = self.registry.create_component('retriever', 'bm25', retriever_config)
        self.add_component('bm25_retriever', retriever)
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from chunks."""
        bm25_retriever = self.get_component('bm25_retriever')
        texts = [chunk.get('content', '') for chunk in self.chunks]
        
        if hasattr(bm25_retriever, 'build_index'):
            bm25_retriever.build_index(texts)
    
    def _combine_results(self, semantic_docs: List[Dict], bm25_docs: List[Dict]) -> List[Dict]:
        """Combine and re-rank results from different retrievers."""
        # Simple combination strategy - you could implement more sophisticated re-ranking
        seen_ids = set()
        combined = []
        
        # Add semantic results with higher weight
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.get('id', i)
            if doc_id not in seen_ids:
                doc['retrieval_score'] = 1.0 - (i * 0.1)  # Higher score for earlier results
                doc['retrieval_method'] = 'semantic'
                combined.append(doc)
                seen_ids.add(doc_id)
        
        # Add BM25 results that weren't already included
        for i, doc in enumerate(bm25_docs):
            doc_id = doc.get('id', f"bm25_{i}")
            if doc_id not in seen_ids:
                doc['retrieval_score'] = 0.8 - (i * 0.1)  # Slightly lower base score
                doc['retrieval_method'] = 'bm25'
                combined.append(doc)
                seen_ids.add(doc_id)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get('retrieval_score', 0), reverse=True)
        
        # Return top results
        return combined[:self.config.retrieval.top_k]
    
    # Inherit other helper methods from DefaultOrchestrator
    def _create_loader(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_loader(self)
    
    def _create_chunker(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_chunker(self)
    
    def _create_embedder(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_embedder(self)
    
    def _create_vectorstore(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_vectorstore(self)
    
    def _create_llm(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_llm(self)
    
    def _create_prompter(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._create_prompter(self)
    
    def _load_documents(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._load_documents(self)
    
    def _chunk_documents(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._chunk_documents(self)
    
    def _embed_and_store(self):
        from rag_engine.core.orchestration import DefaultOrchestrator
        DefaultOrchestrator._embed_and_store(self)


class MultiModalOrchestrator(BaseOrchestrator):
    """Orchestrator for handling multiple content types (text, images, etc.)."""
    
    def build(self) -> None:
        """Build multi-modal pipeline."""
        print("🚀 Building multi-modal RAG pipeline...")
        
        try:
            # Create specialized components for different modalities
            self._create_text_pipeline()
            self._create_image_pipeline()
            self._create_multimodal_retriever()
            self._create_multimodal_llm()
            
            # Process different content types
            self._process_multimodal_documents()
            
            self._is_built = True
            print("✅ Multi-modal pipeline built successfully")
            
        except Exception as e:
            print(f"❌ Multi-modal pipeline build failed: {e}")
            raise
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute multi-modal query."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        
        try:
            # Determine query modality and route accordingly
            query_type = self._analyze_query_type(query, kwargs)
            
            if query_type == "text":
                return self._handle_text_query(query, **kwargs)
            elif query_type == "image":
                return self._handle_image_query(query, **kwargs)
            else:
                return self._handle_multimodal_query(query, **kwargs)
                
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get multi-modal orchestrator status."""
        return {
            "orchestrator": "MultiModalOrchestrator",
            "is_built": self._is_built,
            "components": list(self.components.keys()),
            "supported_modalities": ["text", "image"],
            "config": {
                "embedding_provider": getattr(self.config.embedding, 'provider', 'default'),
                "vectorstore_provider": getattr(self.config.vectorstore, 'provider', 'default'),
                "llm_provider": getattr(self.config.llm, 'provider', 'default')
            }
        }
    
    def _create_text_pipeline(self):
        """Create standard text processing pipeline."""
        # Implementation would create text-specific components
        pass
    
    def _create_image_pipeline(self):
        """Create image processing pipeline."""
        # Implementation would create image-specific components
        pass
    
    def _create_multimodal_retriever(self):
        """Create retriever that can handle multiple modalities."""
        pass
    
    def _create_multimodal_llm(self):
        """Create LLM that can handle multiple modalities."""
        pass
    
    def _process_multimodal_documents(self):
        """Process documents of different types."""
        pass
    
    def _analyze_query_type(self, query: str, kwargs: Dict) -> str:
        """Analyze query to determine modality."""
        if "image" in kwargs or any(ext in query.lower() for ext in ['.jpg', '.png', '.jpeg']):
            return "image"
        return "text"
    
    def _handle_text_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle text-only queries."""
        return {"answer": "Text processing not yet implemented", "status": "success"}
    
    def _handle_image_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle image-related queries."""
        return {"answer": "Image processing not yet implemented", "status": "success"}
    
    def _handle_multimodal_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle queries involving multiple modalities."""
        return {"answer": "Multi-modal processing not yet implemented", "status": "success"}


# Register alternative orchestrators
OrchestratorFactory.register_orchestrator("hybrid", HybridRetrievalOrchestrator)
OrchestratorFactory.register_orchestrator("multimodal", MultiModalOrchestrator)
