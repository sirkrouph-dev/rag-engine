"""Pipeline orchestrates the complete RAG workflow."""

import os
import json
from typing import List, Dict, Any
from rag_engine.config.schema import RAGConfig
from rag_engine.core.loader import LOADER_REGISTRY
from rag_engine.core.chunker import DefaultChunker
from rag_engine.core.embedder import DefaultEmbedder
from rag_engine.core.vectorstore import DefaultVectorStore
from rag_engine.core.retriever import DefaultRetriever
from rag_engine.core.llm import DefaultLLM
from rag_engine.core.prompting import DefaultPrompter


class Pipeline:
    """Main RAG pipeline that orchestrates all components."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        
        # Initialize components based on config
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components based on configuration."""
        # Initialize chunker
        self.chunker = DefaultChunker()
        
        # Initialize embedder
        self.embedder = DefaultEmbedder()
        
        # Initialize vector store
        self.vectorstore = DefaultVectorStore()
        
        # Initialize retriever
        self.retriever = DefaultRetriever()
        
        # Initialize LLM
        self.llm = DefaultLLM()
        
        # Initialize prompt template
        self.prompter = DefaultPrompter()
    
    def build(self):
        """Build the RAG pipeline: load, chunk, embed, and store documents."""
        print("🚀 Starting RAG pipeline build...")
        
        # Step 1: Load documents
        self._load_documents()
        
        # Step 2: Chunk documents
        self._chunk_documents()
        
        # Step 3: Generate embeddings and store in vector database
        self._embed_and_store()
        
        print("✅ Pipeline build complete!")
        print(f"   📄 Loaded {len(self.documents)} documents")
        print(f"   🧩 Created {len(self.chunks)} chunks")
        print(f"   💾 Stored in {self.config.vectorstore.provider} vector store")
    
    def _load_documents(self):
        """Load all documents specified in configuration."""
        print("📂 Loading documents...")
        self.documents = []
        
        for doc_config in self.config.documents:
            doc_type = doc_config.type.lower()
            doc_path = doc_config.path
            
            if not os.path.exists(doc_path):
                print(f"⚠️  Warning: Document not found: {doc_path}")
                continue
            
            if doc_type not in LOADER_REGISTRY:
                print(f"⚠️  Warning: Unknown document type: {doc_type}")
                continue
            
            loader = LOADER_REGISTRY[doc_type]
            try:
                docs = loader.load({"path": doc_path, "type": doc_type})
                self.documents.extend(docs)
                print(f"   ✓ Loaded {doc_path} ({len(docs)} items)")
            except Exception as e:                print(f"   ✗ Failed to load {doc_path}: {str(e)}")
    
    def _chunk_documents(self):
        """Chunk all loaded documents."""
        print("🧩 Chunking documents...")
        
        chunk_config = {
            "method": self.config.chunking.method,
            "max_tokens": self.config.chunking.max_tokens,
            "overlap": self.config.chunking.overlap,
            "chunk_size": self.config.chunking.max_tokens * 4,  # Rough chars per token
            "chunk_overlap": self.config.chunking.overlap * 4
        }
        
        try:
            self.chunks = self.chunker.chunk(self.documents, chunk_config)
            print(f"   ✓ Created {len(self.chunks)} chunks")
        except Exception as e:
            print(f"   ✗ Failed to chunk documents: {str(e)}")
            self.chunks = []
    
    def _embed_and_store(self):
        """Generate embeddings and store in vector database."""
        print("🔢 Generating embeddings and storing in vector database...")        # Initialize vector store
        vectorstore_config = {
            "provider": self.config.vectorstore.provider,
            "persist_directory": self.config.vectorstore.persist_directory
        }
          # Generate embeddings for all chunks
        embedding_config = {
            "provider": self.config.embedding.provider,
            "model": self.config.embedding.model,
            "api_key": self.config.embedding.api_key
        }
        
        try:
            # Embed all chunks at once
            embedded_chunks = self.embedder.embed(self.chunks, embedding_config)
            
            # Store chunks in vector database
            self.vectorstore.add(embedded_chunks, vectorstore_config)
            
            # Persist vector store
            self.vectorstore.persist(vectorstore_config)
            print(f"   ✓ Embedded and stored {len(embedded_chunks)} chunks")
            print(f"   ✓ Vector store persisted to {self.config.vectorstore.persist_directory}")
            
        except Exception as e:
            print(f"   ✗ Failed to embed and store chunks: {str(e)}")
            raise
    
    def chat(self, query: str = None):
        """Interactive chat interface with the RAG system."""
        print("💬 Starting chat mode...")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("-" * 50)
          # Load existing vector store if it exists
        if not self.vectorstore:
            print("❌ No vector store found. Please run 'build' first.")
            return
        
        try:
            vectorstore_config = {
                "provider": self.config.vectorstore.provider,
                "persist_directory": self.config.vectorstore.persist_directory
            }
            self.vectorstore.load(vectorstore_config)
        except Exception as e:
            print(f"❌ Failed to load vector store: {str(e)}")
            return
        
        # Start chat loop
        while True:
            if query is None:
                user_input = input("\n🤔 Your question: ").strip()
            else:
                user_input = query
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = self._generate_response(user_input)
                print(f"\n🤖 Answer: {response}")
            except Exception as e:
                print(f"❌ Error generating response: {str(e)}")
            
            if query is not None:  # Single query mode
                break
    
    def _generate_response(self, query: str) -> str:
        """Generate response for a user query using RAG."""
        # Step 1: Retrieve relevant chunks
        retrieval_config = {
            "retrieval_strategy": "simple",
            "top_k": self.config.retrieval.top_k,
            "embedder": self.embedder,
            "embedding_config": {
                "provider": "openai",
                "model": self.config.embedding.model,
                "api_key": self.config.embedding.api_key
            }
        }
        
        try:
            relevant_chunks = self.retriever.retrieve(
                self.vectorstore, query, retrieval_config
            )
        except Exception as e:
            print(f"Warning: Retrieval failed: {str(e)}")
            relevant_chunks = []
        
        # Step 2: Build context from retrieved chunks
        context = "\n\n".join([chunk.get("content", "") for chunk in relevant_chunks])
        
        # Step 3: Generate prompt
        prompt_config = {
            "system_prompt": self.config.prompting.system_prompt,
            "context": context,
            "query": query
        }
        
        prompt = self.prompter.build_prompt(prompt_config)
        
        # Step 4: Generate response using LLM
        llm_config = {
            "provider": self.config.llm.provider,
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "api_key": getattr(self.config.embedding, 'api_key', None)  # Use embedding API key for now
        }
        
        response = self.llm.generate(prompt, llm_config)
        
        return response
