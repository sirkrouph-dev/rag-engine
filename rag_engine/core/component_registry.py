"""
Default component registration for the RAG Engine.

This module registers all built-in component implementations with the
global component registry.
"""

from rag_engine.core.orchestration import COMPONENT_REGISTRY
from rag_engine.core.loader import LOADER_REGISTRY
from rag_engine.core.chunker import get_chunker, DefaultChunker
from rag_engine.core.embedder import get_embedder, DefaultEmbedder  
from rag_engine.core.vectorstore import get_vector_store, DefaultVectorStore
from rag_engine.core.retriever import get_retriever, DefaultRetriever
from rag_engine.core.llm import get_llm, DefaultLLM
from rag_engine.core.prompting import get_prompter, DefaultPrompter
from rag_engine.core.prompting_enhanced import get_prompter as get_enhanced_prompter


def register_default_components():
    """Register all default component implementations."""
    
    # Register loaders
    for name, loader_class in LOADER_REGISTRY.items():
        COMPONENT_REGISTRY.register_component(
            'loader',
            name,
            loader_class,
            f"Document loader: {name}"
        )
    
    # Register chunker factory
    def chunker_factory(name: str, config: dict):
        return get_chunker(config)
    
    COMPONENT_REGISTRY.register_factory('chunker', chunker_factory)
    
    # Register chunker implementations
    chunker_methods = ['fixed_size', 'sentence', 'token']
    for method in chunker_methods:
        COMPONENT_REGISTRY.register_component(
            'chunker',
            method,
            DefaultChunker,
            f"Text chunker: {method}"
        )
    
    # Register embedder factory
    def embedder_factory(name: str, config: dict):
        return get_embedder(config)
    
    COMPONENT_REGISTRY.register_factory('embedder', embedder_factory)
    
    # Register embedder implementations
    embedder_providers = ['huggingface', 'openai', 'local']
    for provider in embedder_providers:
        COMPONENT_REGISTRY.register_component(
            'embedder',
            provider,
            DefaultEmbedder,
            f"Embedding provider: {provider}"
        )
      # Register vectorstore factory
    def vectorstore_factory(name: str, config: dict):
        return get_vector_store(config)
    
    COMPONENT_REGISTRY.register_factory('vectorstore', vectorstore_factory)
    
    # Register vectorstore implementations
    vectorstore_providers = ['chroma', 'faiss', 'pinecone']
    for provider in vectorstore_providers:
        COMPONENT_REGISTRY.register_component(
            'vectorstore',
            provider,
            DefaultVectorStore,
            f"Vector store: {provider}"
        )
    
    # Register retriever factory
    def retriever_factory(name: str, config: dict):
        return get_retriever(config)
    
    COMPONENT_REGISTRY.register_factory('retriever', retriever_factory)
    
    # Register retriever implementations
    retriever_methods = ['similarity', 'bm25', 'hybrid', 'mmr']
    for method in retriever_methods:
        COMPONENT_REGISTRY.register_component(
            'retriever',
            method,
            DefaultRetriever,
            f"Retrieval method: {method}"
        )
    
    # Register LLM factory
    def llm_factory(name: str, config: dict):
        return get_llm(config)
    
    COMPONENT_REGISTRY.register_factory('llm', llm_factory)
    
    # Register LLM implementations
    llm_providers = ['openai', 'anthropic', 'local', 'ollama']
    for provider in llm_providers:
        COMPONENT_REGISTRY.register_component(
            'llm',
            provider,
            DefaultLLM,
            f"LLM provider: {provider}"
        )
    
    # Register prompter factory
    def prompter_factory(name: str, config: dict):
        # Try enhanced prompter first, fall back to legacy
        try:
            return get_enhanced_prompter(name, config)
        except (ValueError, NotImplementedError):
            return get_prompter(config)
    
    COMPONENT_REGISTRY.register_factory('prompter', prompter_factory)
    
    # Register prompter implementations (enhanced + legacy)
    prompter_templates = [
        'default', 'conversational', 'qa', 'summarization',  # Legacy
        'rag', 'code_explanation', 'debugging', 'chain_of_thought'  # Enhanced
    ]
    for template in prompter_templates:
        COMPONENT_REGISTRY.register_component(
            'prompter',
            template,
            DefaultPrompter,
            f"Prompt template: {template}"
        )
    
    print("📦 Registered all default components")


# Auto-register components when module is imported
register_default_components()
