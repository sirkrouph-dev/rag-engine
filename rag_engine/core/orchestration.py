"""
Modular orchestration layer for the RAG Engine.

This module provides a flexible architecture for composing and managing
different RAG components (retrievers, LLMs, embedders, etc.) without
tight coupling between the API layer and specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Callable
from dataclasses import dataclass
import importlib
from pathlib import Path

from rag_engine.config.schema import RAGConfig


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: str
    class_ref: Type
    description: str
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None


class ComponentRegistry:
    """Registry for managing different component implementations."""
    
    def __init__(self):
        self._components: Dict[str, Dict[str, ComponentInfo]] = {
            'loader': {},
            'chunker': {},
            'embedder': {},
            'vectorstore': {},
            'retriever': {},
            'llm': {},
            'prompter': {},
            'reasoner': {},
            'tool': {}
        }
        self._factories: Dict[str, Callable] = {}
    
    def register_component(
        self,
        component_type: str,
        name: str,
        class_ref: Type,
        description: str = "",
        dependencies: List[str] = None,
        config_schema: Dict[str, Any] = None
    ) -> None:
        """Register a component implementation."""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        info = ComponentInfo(
            name=name,
            component_type=component_type,
            class_ref=class_ref,
            description=description,
            dependencies=dependencies or [],
            config_schema=config_schema or {}
        )
        
        self._components[component_type][name] = info
        print(f"📦 Registered {component_type}: {name}")
    
    def register_factory(self, component_type: str, factory_func: Callable) -> None:
        """Register a factory function for creating components."""
        self._factories[component_type] = factory_func
    
    def get_component_info(self, component_type: str, name: str) -> Optional[ComponentInfo]:
        """Get information about a specific component."""
        return self._components.get(component_type, {}).get(name)
    
    def list_components(self, component_type: str = None) -> Dict[str, List[str]]:
        """List all registered components, optionally filtered by type."""
        if component_type:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        
        return {
            comp_type: list(components.keys())
            for comp_type, components in self._components.items()
        }
    
    def create_component(self, component_type: str, name: str, config: Dict[str, Any] = None):
        """Create a component instance using its factory or class."""
        component_info = self.get_component_info(component_type, name)
        if not component_info:
            raise ValueError(f"Component {component_type}:{name} not found")
        
        # Use factory if available, otherwise use class constructor
        if component_type in self._factories:
            return self._factories[component_type](name, config or {})
        else:
            return component_info.class_ref(**(config or {}))
    
    def load_plugin(self, plugin_path: str) -> None:
        """Load a plugin module that registers additional components."""
        try:
            if plugin_path.endswith('.py'):
                # Load from file path
                spec = importlib.util.spec_from_file_location("plugin", plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Load from module name
                module = importlib.import_module(plugin_path)
            
            # Call register function if it exists
            if hasattr(module, 'register_components'):
                module.register_components(self)
                print(f"🔌 Loaded plugin: {plugin_path}")
        except Exception as e:
            print(f"❌ Failed to load plugin {plugin_path}: {e}")


class BaseOrchestrator(ABC):
    """Abstract base class for orchestrating RAG components."""
    
    def __init__(self, config: RAGConfig, registry: ComponentRegistry):
        self.config = config
        self.registry = registry
        self.components = {}
        self._is_built = False
    
    @abstractmethod
    def build(self) -> None:
        """Build and configure all components."""
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a query through the orchestrated pipeline."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the orchestrator."""
        pass
    
    def get_component(self, component_type: str):
        """Get a component instance."""
        return self.components.get(component_type)
    
    def add_component(self, component_type: str, instance: Any) -> None:
        """Add a component instance to the orchestrator."""
        self.components[component_type] = instance


class DefaultOrchestrator(BaseOrchestrator):
    """Default orchestrator implementing standard RAG pipeline."""
    
    def build(self) -> None:
        """Build the standard RAG pipeline."""
        print("🚀 Building RAG pipeline with DefaultOrchestrator...")
        
        try:
            # Create components based on configuration
            self._create_loader()
            self._create_chunker()
            self._create_embedder()
            self._create_vectorstore()
            self._create_retriever()
            self._create_llm()
            self._create_prompter()
            
            # Load and process documents
            self._load_documents()
            self._chunk_documents()
            self._embed_and_store()
            
            self._is_built = True
            print("✅ Pipeline built successfully")
            
        except Exception as e:
            print(f"❌ Pipeline build failed: {e}")
            raise
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a query through the pipeline."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call build() first.")
        
        try:
            # 1. Retrieve relevant documents
            retriever = self.get_component('retriever')
            retrieved_docs = retriever.retrieve(query, **kwargs)
            
            # 2. Generate prompt
            prompter = self.get_component('prompter')
            prompt = prompter.generate_prompt(query, retrieved_docs)
            
            # 3. Generate response
            llm = self.get_component('llm')
            response = llm.generate(prompt)
            
            return {
                "answer": response,
                "sources": [doc.get("metadata", {}) for doc in retrieved_docs],
                "retrieved_docs": len(retrieved_docs),
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
            "orchestrator": "DefaultOrchestrator",
            "is_built": self._is_built,
            "components": list(self.components.keys()),
            "config": {
                "embedding_provider": getattr(self.config.embedding, 'provider', 'default'),
                "vectorstore_provider": getattr(self.config.vectorstore, 'provider', 'default'),
                "llm_provider": getattr(self.config.llm, 'provider', 'default')
            }
        }
    
    def _create_loader(self) -> None:
        """Create document loader component."""
        loader_config = getattr(self.config, 'loader', {})
        provider = getattr(loader_config, 'provider', 'default')
        
        loader = self.registry.create_component('loader', provider, loader_config.__dict__ if hasattr(loader_config, '__dict__') else {})
        self.add_component('loader', loader)
    
    def _create_chunker(self) -> None:
        """Create chunker component."""
        chunker_config = self.config.chunking
        method = getattr(chunker_config, 'method', 'fixed_size')
        
        chunker = self.registry.create_component('chunker', method, chunker_config.__dict__)
        self.add_component('chunker', chunker)
    
    def _create_embedder(self) -> None:
        """Create embedder component."""
        embedder_config = self.config.embedding
        provider = getattr(embedder_config, 'provider', 'huggingface')
        
        embedder = self.registry.create_component('embedder', provider, embedder_config.__dict__)
        self.add_component('embedder', embedder)
    
    def _create_vectorstore(self) -> None:
        """Create vector store component."""
        vectorstore_config = self.config.vectorstore
        provider = getattr(vectorstore_config, 'provider', 'chroma')
        
        vectorstore = self.registry.create_component('vectorstore', provider, vectorstore_config.__dict__)
        self.add_component('vectorstore', vectorstore)
    
    def _create_retriever(self) -> None:
        """Create retriever component."""
        retrieval_config = self.config.retrieval
        method = getattr(retrieval_config, 'method', 'similarity')
        
        # Pass vectorstore to retriever
        vectorstore = self.get_component('vectorstore')
        retriever_config = retrieval_config.__dict__.copy()
        retriever_config['vectorstore'] = vectorstore
        
        retriever = self.registry.create_component('retriever', method, retriever_config)
        self.add_component('retriever', retriever)
    
    def _create_llm(self) -> None:
        """Create LLM component."""
        llm_config = self.config.llm
        provider = getattr(llm_config, 'provider', 'openai')
        
        llm = self.registry.create_component('llm', provider, llm_config.__dict__)
        self.add_component('llm', llm)
    
    def _create_prompter(self) -> None:
        """Create prompter component."""
        prompt_config = getattr(self.config, 'prompting', {})
        template = getattr(prompt_config, 'template', 'default')
        
        prompter = self.registry.create_component('prompter', template, prompt_config.__dict__ if hasattr(prompt_config, '__dict__') else {})
        self.add_component('prompter', prompter)
    
    def _load_documents(self) -> None:
        """Load documents using the loader component."""
        loader = self.get_component('loader')
        documents = []
        
        for doc_config in self.config.documents:
            docs = loader.load(doc_config.path)
            documents.extend(docs)
        
        self.documents = documents
        print(f"📄 Loaded {len(documents)} documents")
    
    def _chunk_documents(self) -> None:
        """Chunk documents using the chunker component."""
        chunker = self.get_component('chunker')
        chunks = []
        
        for doc in self.documents:
            doc_chunks = chunker.chunk(doc.get('content', ''), doc.get('metadata', {}))
            chunks.extend(doc_chunks)
        
        self.chunks = chunks
        print(f"✂️  Created {len(chunks)} chunks")
    
    def _embed_and_store(self) -> None:
        """Embed chunks and store in vector database."""
        embedder = self.get_component('embedder')
        vectorstore = self.get_component('vectorstore')
        
        # Extract text and metadata
        texts = [chunk.get('content', '') for chunk in self.chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in self.chunks]
        
        # Generate embeddings
        embeddings = embedder.embed_documents(texts)
        
        # Store in vector database
        vectorstore.add_documents(texts, embeddings, metadatas)
        print(f"🔮 Embedded and stored {len(texts)} chunks")


class OrchestratorFactory:
    """Factory for creating orchestrator instances."""
    
    _orchestrators = {}
    
    @classmethod
    def register_orchestrator(cls, name: str, orchestrator_class: Type[BaseOrchestrator]):
        """Register an orchestrator implementation."""
        cls._orchestrators[name] = orchestrator_class
    
    @classmethod
    def create_orchestrator(
        cls,
        name: str,
        config: RAGConfig,
        registry: ComponentRegistry
    ) -> BaseOrchestrator:
        """Create an orchestrator instance."""
        if name not in cls._orchestrators:
            raise ValueError(f"Unknown orchestrator: {name}. Available: {list(cls._orchestrators.keys())}")
        
        orchestrator_class = cls._orchestrators[name]
        return orchestrator_class(config, registry)
    
    @classmethod
    def list_orchestrators(cls) -> List[str]:
        """List all registered orchestrators."""
        return list(cls._orchestrators.keys())


# Global registry instance
COMPONENT_REGISTRY = ComponentRegistry()

# Register default orchestrator
OrchestratorFactory.register_orchestrator("default", DefaultOrchestrator)


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return COMPONENT_REGISTRY


def create_orchestrator(
    orchestrator_type: str = "default",
    config: RAGConfig = None,
    registry: ComponentRegistry = None
) -> BaseOrchestrator:
    """Convenience function to create an orchestrator."""
    if registry is None:
        registry = get_global_registry()
    
    return OrchestratorFactory.create_orchestrator(orchestrator_type, config, registry)
