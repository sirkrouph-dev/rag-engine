# Embedder interface and implementations

import os
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional
from abc import ABC
import importlib
from rag_engine.core.base import BaseEmbedder

logger = logging.getLogger(__name__)

class EmbedProvider(ABC):
    """Base class for different embedding providers."""
    def embed_query(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embeddings for a single query text."""
        raise NotImplementedError
        
    def embed_documents(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings for multiple document texts."""
        raise NotImplementedError


class OpenAIEmbedProvider(EmbedProvider):
    """Generates embeddings using OpenAI's embedding models."""
    def __init__(self):
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'")
            raise
            
    def _get_client(self, config: Dict[str, Any]):
        """Get an initialized OpenAI client."""
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or environment")
        return self.OpenAI(api_key=api_key)
    
    def _get_model(self, config: Dict[str, Any]) -> str:
        """Get the embedding model name."""
        return config.get("model", "text-embedding-3-large")
    
    def embed_query(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embeddings for a single query text."""
        client = self._get_client(config)
        model = self._get_model(config)
        
        # For text-embedding-3-* models, dimensions can be specified
        dimensions = config.get("dimensions", None)
        extra_args = {"dimensions": dimensions} if dimensions else {}
        
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float",
            **extra_args
        )
        
        return response.data[0].embedding
        
    def embed_documents(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings for multiple document texts."""
        client = self._get_client(config)
        model = self._get_model(config)
        
        # Handle large batches with limits (OpenAI has token limits per request)
        batch_size = config.get("batch_size", 100)  # Default batch of 100 texts
        dimensions = config.get("dimensions", None)
        extra_args = {"dimensions": dimensions} if dimensions else {}
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch_texts,
                encoding_format="float",
                **extra_args
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings


class GeminiVertexEmbedProvider(EmbedProvider):
    """Generates embeddings using Google's Gemini/Vertex AI embedding models."""
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
            self.use_vertex = False
        except ImportError:
            logger.error("Google GenerativeAI package not installed. Run 'pip install google-generativeai'")
            raise
            
        # Check if Vertex AI is available
        try:
            from google.cloud import aiplatform
            self.aiplatform = aiplatform
            self.use_vertex = True
        except ImportError:
            logger.info("Google Cloud AI Platform not installed. Vertex AI will not be available.")
            self.use_vertex = False
    
    def _setup_gemini(self, config: Dict[str, Any]):
        """Set up Gemini client."""
        api_key = config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided in config or environment")
        self.genai.configure(api_key=api_key)
    
    def _setup_vertex(self, config: Dict[str, Any]):
        """Set up Vertex AI client."""
        project = config.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = config.get("location", "us-central1")
        
        if not project:
            raise ValueError("GCP project ID not provided in config or environment")
            
        self.aiplatform.init(project=project, location=location)
    
    def embed_query(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embeddings for a single query text."""
        if config.get("use_vertex", self.use_vertex) and self.use_vertex:
            return self._embed_query_vertex(text, config)
        else:
            return self._embed_query_gemini(text, config)
    
    def _embed_query_gemini(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embedding using Gemini API."""
        self._setup_gemini(config)
        model = config.get("model", "models/embedding-001")
        
        result = self.genai.embed_content(
            model=model,
            content=text,
            task_type=config.get("task_type", "retrieval_query")
        )
        
        return result["embedding"]
    
    def _embed_query_vertex(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embedding using Vertex AI."""
        self._setup_vertex(config)
        
        model_name = config.get("model", "textembedding-gecko@001")
        endpoint = self.aiplatform.Endpoint(f"projects/{config['project']}/locations/{config['location']}/publishers/google/models/{model_name}")
        
        response = endpoint.predict(instances=[{"content": text}])
        return response.predictions[0]["embeddings"]["values"]
    
    def embed_documents(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings for multiple document texts."""
        if config.get("use_vertex", self.use_vertex) and self.use_vertex:
            return self._embed_documents_vertex(texts, config)
        else:
            return self._embed_documents_gemini(texts, config)
    
    def _embed_documents_gemini(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using Gemini API."""
        self._setup_gemini(config)
        model = config.get("model", "models/embedding-001")
        batch_size = config.get("batch_size", 100)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Gemini API doesn't support true batching yet, so we make individual calls
            batch_embeddings = []
            for text in batch_texts:
                result = self.genai.embed_content(
                    model=model,
                    content=text,
                    task_type=config.get("task_type", "retrieval_document")
                )
                batch_embeddings.append(result["embedding"])
                
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def _embed_documents_vertex(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using Vertex AI."""
        self._setup_vertex(config)
        
        model_name = config.get("model", "textembedding-gecko@001")
        endpoint = self.aiplatform.Endpoint(f"projects/{config['project']}/locations/{config['location']}/publishers/google/models/{model_name}")
        
        batch_size = config.get("batch_size", 100)  # Vertex may have batch size limits
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            instances = [{"content": text} for text in batch_texts]
            
            response = endpoint.predict(instances=instances)
            batch_embeddings = [pred["embeddings"]["values"] for pred in response.predictions]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings


class HuggingFaceEmbedProvider(EmbedProvider):
    """Generates embeddings using Hugging Face models via SentenceTransformers."""
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.SentenceTransformer = SentenceTransformer
        except ImportError:
            logger.error("SentenceTransformers package not installed. Run 'pip install sentence-transformers'")
            raise
            
        self.model = None
        self.current_model_name = None
    
    def _get_model(self, config: Dict[str, Any]):
        """Get the embedding model, loading if necessary."""
        model_name = config.get("model", "sentence-transformers/all-mpnet-base-v2")
        
        if self.model is None or self.current_model_name != model_name:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = self.SentenceTransformer(model_name)
            self.current_model_name = model_name
            
        return self.model
    
    def embed_query(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Generate embeddings for a single query text."""
        model = self._get_model(config)
        embeddings = model.encode([text], normalize_embeddings=config.get("normalize", True))
        return embeddings[0].tolist()
    
    def embed_documents(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings for multiple document texts."""
        model = self._get_model(config)
        batch_size = config.get("batch_size", 32)  # Default batch size for transformers
        
        # Set show_progress_bar based on batch size and text count
        show_progress = len(texts) > 100
        
        embeddings = model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=config.get("normalize", True),
            show_progress_bar=show_progress
        )
        
        return embeddings.tolist()


class DefaultEmbedder(BaseEmbedder):
    """Main embedder class that delegates to appropriate provider based on config."""
    def __init__(self):
        self.providers = {
            "openai": OpenAIEmbedProvider(),
            "gemini": GeminiVertexEmbedProvider(),
            "huggingface": HuggingFaceEmbedProvider()
        }
    
    def _get_provider(self, config: Dict[str, Any]) -> EmbedProvider:
        """Get the configured embedding provider."""
        provider_name = config.get("provider", "openai").lower()
        
        if provider_name not in self.providers:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
            
        return self.providers[provider_name]
    
    def embed(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Embed document chunks and return with embeddings added."""
        provider = self._get_provider(config)
        
        # Extract text content from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings for all texts
        embeddings = provider.embed_documents(texts, config)
        
        # Add embeddings back to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
            
        return chunks
        
    def embed_query(self, query: str, config: Dict[str, Any]) -> List[float]:
        """Embed a query text and return the embedding."""
        provider = self._get_provider(config)
        return provider.embed_query(query, config)


# Factory function to get the appropriate embedder
def get_embedder(config: Dict[str, Any]) -> BaseEmbedder:
    return DefaultEmbedder()
