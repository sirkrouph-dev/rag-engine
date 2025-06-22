"""Prompting interface and implementations for RAG Engine."""

from typing import Dict, Any
from rag_engine.core.base import BasePrompting


class DefaultPrompter(BasePrompting):
    """Default prompt template builder for RAG."""
    
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a complete prompt from system prompt, context, and query."""
        system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        context = config.get("context", "")
        query = config.get("query", "")
        
        # Build the complete prompt
        if context:
            prompt = f"""System: {system_prompt}

Context:
{context}

User: {query}"""
        else:
            prompt = f"""System: {system_prompt}

User: {query}"""

        return prompt
    
    def format(self, context, config):
        """Legacy method for compatibility with base class."""
        return self.build_prompt({
            "system_prompt": config.get("system_prompt", ""),
            "context": context,
            "query": config.get("query", "")
        })

def get_prompter(config: Dict[str, Any]) -> BasePrompting:
    """Factory function to get the appropriate prompter."""
    template_type = config.get("template", "default")
    
    # For now, just return DefaultPrompter regardless of template type
    # In the future, different template types could return different implementations
    return DefaultPrompter()
