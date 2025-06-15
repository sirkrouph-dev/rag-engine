# LLM interface and implementations

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC
import importlib
from rag_engine.core.base import BaseLLM

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Base abstract class for different LLM providers."""
    def generate(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate text based on prompt using the configured LLM."""
        raise NotImplementedError
        
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        """Stream responses from the LLM. Returns a generator."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using OpenAI API."""
    def __init__(self):
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'")
            raise
            
    def generate(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate text using OpenAI models."""
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or environment")
            
        model = config.get("model", "gpt-4")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        
        client = self.OpenAI(api_key=api_key)
        
        # Format system prompt and user message
        messages = []
        if "system_prompt" in config:
            messages.append({"role": "system", "content": config["system_prompt"]})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stream=False
        )
        
        return response.choices[0].message.content
        
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        """Stream responses from OpenAI."""
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or environment")
            
        model = config.get("model", "gpt-4")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        
        client = self.OpenAI(api_key=api_key)
        
        # Format system prompt and user message
        messages = []
        if "system_prompt" in config:
            messages.append({"role": "system", "content": config["system_prompt"]})
        messages.append({"role": "user", "content": prompt})
        
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""
    def __init__(self):
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            logger.error("Google Generative AI package not installed. Run 'pip install google-generativeai'")
            raise
            
    def generate(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate text using Google Gemini models."""
        api_key = config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided in config or environment")
            
        model_name = config.get("model", "gemini-1.5-pro")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        
        self.genai.configure(api_key=api_key)
        model = self.genai.GenerativeModel(model_name=model_name)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": config.get("top_p", 1.0),
            "top_k": config.get("top_k", 40)
        }
        
        # Format prompt with system prompt if provided
        if "system_prompt" in config:
            prompt = f"{config['system_prompt']}\n\n{prompt}"
            
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
        
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        """Stream responses from Gemini."""
        api_key = config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided in config or environment")
            
        model_name = config.get("model", "gemini-1.5-pro")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 1000)
        
        self.genai.configure(api_key=api_key)
        model = self.genai.GenerativeModel(model_name=model_name)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": config.get("top_p", 1.0),
            "top_k": config.get("top_k", 40)
        }
        
        # Format prompt with system prompt if provided
        if "system_prompt" in config:
            prompt = f"{config['system_prompt']}\n\n{prompt}"
            
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class LocalModelProvider(LLMProvider):
    """Provider for local models like Phi-3 and Gemma using Transformers or Ollama."""
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        self.current_model = None
        
    def _load_transformers_model(self, model_id, config):
        """Load model using Transformers library."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            logger.error("Transformers package not installed. Run 'pip install transformers torch'")
            raise
            
        # Determine device - CUDA, MPS (Mac), or CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        logger.info(f"Loading model {model_id} on {device}")
        
        # Configure model loading parameters based on available resources
        load_in_8bit = config.get("load_in_8bit", False)
        load_in_4bit = config.get("load_in_4bit", False)
        
        # Load the model and tokenizer
        model_kwargs = {}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["device_map"] = device
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device if isinstance(device, int) else 0 if device.type == "cuda" else -1
        )
        
        self.model_loaded = True
        self.current_model = model_id
        
    def _load_ollama_model(self, model_name, config):
        """Load model via Ollama server."""
        try:
            import ollama
        except ImportError:
            logger.error("Ollama package not installed. Run 'pip install ollama'")
            raise
            
        # Just verify the model exists in Ollama
        ollama_host = config.get("ollama_host", "http://localhost:11434")
        available_models = ollama.list(host=ollama_host)
        
        model_exists = False
        for model in available_models.get('models', []):
            if model.get('name') == model_name:
                model_exists = True
                break
                
        if not model_exists:
            logger.warning(f"Model {model_name} not found in Ollama. Attempting to pull.")
            ollama.pull(model_name, host=ollama_host)
            
        self.model_loaded = True
        self.current_model = model_name
        
    def _get_model_config(self, config: Dict[str, Any]):
        """Extract model-specific configuration."""
        model_name = config.get("model")
        model_provider = config.get("model_provider", "transformers")
        
        if not model_name:
            raise ValueError("Model name must be provided for local models")
            
        return model_name, model_provider
        
    def generate(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate text using local models."""
        model_name, model_provider = self._get_model_config(config)
        
        # Load model if not already loaded or if different model requested
        if not self.model_loaded or self.current_model != model_name:
            if model_provider == "transformers":
                self._load_transformers_model(model_name, config)
            elif model_provider == "ollama":
                self._load_ollama_model(model_name, config)
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
                
        # Generate using transformers pipeline
        if model_provider == "transformers":
            max_length = config.get("max_tokens", 1000)
            temperature = config.get("temperature", 0.7)
            
            # Format prompt with system prompt if provided
            if "system_prompt" in config:
                full_prompt = f"{config['system_prompt']}\n\n{prompt}"
            else:
                full_prompt = prompt
                
            # Generate text
            result = self.pipeline(
                full_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=config.get("top_p", 0.9),
                num_return_sequences=1,
                eos_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            # Remove the prompt from the beginning of the response
            response = generated_text[len(full_prompt):].strip()
            return response
            
        # Generate using Ollama
        elif model_provider == "ollama":
            import ollama
            
            ollama_host = config.get("ollama_host", "http://localhost:11434")
            
            # Format prompt with system prompt if provided
            system_prompt = config.get("system_prompt", "")
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                system=system_prompt,
                options={
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 0.9),
                    "top_k": config.get("top_k", 40),
                    "num_predict": config.get("max_tokens", 1000)
                },
                host=ollama_host
            )
            
            return response["response"]
            
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        """Stream responses from local models."""
        model_name, model_provider = self._get_model_config(config)
        
        # Load model if not already loaded or if different model requested
        if not self.model_loaded or self.current_model != model_name:
            if model_provider == "transformers":
                self._load_transformers_model(model_name, config)
            elif model_provider == "ollama":
                self._load_ollama_model(model_name, config)
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
                
        # Stream using Ollama (Transformers streaming is complex, using Ollama is easier)
        if model_provider == "ollama":
            import ollama
            
            ollama_host = config.get("ollama_host", "http://localhost:11434")
            system_prompt = config.get("system_prompt", "")
            
            stream = ollama.generate(
                model=model_name,
                prompt=prompt,
                system=system_prompt,
                options={
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 0.9),
                    "top_k": config.get("top_k", 40),
                    "num_predict": config.get("max_tokens", 1000)
                },
                stream=True,
                host=ollama_host
            )
            
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
        else:
            # For transformers, we generate the full response and yield it in one go
            # as streaming with transformers is complex
            yield self.generate(prompt, config)


class DefaultLLM(BaseLLM):
    """Main LLM class that delegates to appropriate provider based on config."""
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "local": LocalModelProvider()
        }
        
    def generate(self, prompt: str, config: Dict[str, Any]) -> str:
        provider_name = config.get("provider", "openai").lower()
        
        if provider_name not in self.providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            
        provider = self.providers[provider_name]
        return provider.generate(prompt, config)
        
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        provider_name = config.get("provider", "openai").lower()
        
        if provider_name not in self.providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            
        provider = self.providers[provider_name]
        return provider.generate_streaming(prompt, config)


# Factory function to get the appropriate LLM
def get_llm(config: Dict[str, Any]) -> BaseLLM:
    return DefaultLLM()
