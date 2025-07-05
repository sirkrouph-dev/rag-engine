"""Enhanced prompting interface and implementations for RAG Engine."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path


class BasePrompter(ABC):
    """Base class for all prompt templates and processors."""
    
    @abstractmethod
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a complete prompt from configuration."""
        pass
    
    @abstractmethod
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format retrieved documents into context."""
        pass


class DefaultPrompter(BasePrompter):
    """Default prompt template builder for RAG."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.context_separator = self.config.get("context_separator", "\n---\n")
        self.include_metadata = self.config.get("include_metadata", False)
    
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
    
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format retrieved documents into context."""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add metadata if enabled
            doc_text = content
            if self.include_metadata and metadata:
                source = metadata.get("source", f"Document {i+1}")
                doc_text = f"[Source: {source}]\n{content}"
            
            # Check length limit
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return self.context_separator.join(context_parts)


class RAGPrompter(BasePrompter):
    """Specialized prompter for RAG applications with advanced context management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.context_window = self.config.get("context_window", 3000)
        self.citation_format = self.config.get("citation_format", "numbered")
        self.template_path = self.config.get("template_path")
        self.custom_template = self._load_template()
    
    def _load_template(self) -> Optional[str]:
        """Load custom template from file if specified."""
        if self.template_path and os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not load template from {self.template_path}: {e}")
        return None
    
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a RAG-optimized prompt."""
        system_prompt = config.get("system_prompt", 
            "You are a helpful AI assistant. Use the provided context to answer questions accurately.")
        context = config.get("context", "")
        query = config.get("query", "")
        
        if self.custom_template:
            return self.custom_template.format(
                system_prompt=system_prompt,
                context=context,
                query=query
            )
        
        # Default RAG template
        if context:
            prompt = f"""{system_prompt}

Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Instructions:
- Base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Be precise and cite relevant parts when appropriate
- Do not make up information not present in the context

Answer:"""
        else:
            prompt = f"""{system_prompt}

Question: {query}

Answer:"""
        
        return prompt
    
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format documents with citation support."""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0.0)
            
            # Add citations based on format
            if self.citation_format == "numbered":
                doc_text = f"[{i}] {content}"
                if metadata.get("source"):
                    doc_text += f" (Source: {metadata['source']})"
            elif self.citation_format == "bracketed":
                source = metadata.get("source", f"doc{i}")
                doc_text = f"[{source}] {content}"
            else:
                doc_text = content
            
            # Check length limit
            if current_length + len(doc_text) > self.context_window:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)


class ConversationalPrompter(BasePrompter):
    """Prompter for multi-turn conversations with memory."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_length = self.config.get("memory_length", 10)
        self.persona = self.config.get("persona", "helpful_assistant")
        self.conversation_history = []
    
    def _get_persona_prompt(self) -> str:
        """Get system prompt based on persona."""
        personas = {
            "helpful_assistant": "You are a helpful and knowledgeable AI assistant.",
            "technical_expert": "You are a technical expert who provides detailed, accurate information.",
            "friendly_tutor": "You are a friendly tutor who explains concepts clearly and patiently.",
            "research_assistant": "You are a research assistant who provides thorough, well-sourced information."
        }
        return personas.get(self.persona, personas["helpful_assistant"])
    
    def _format_history(self) -> str:
        """Format conversation history for inclusion in prompt."""
        if not self.conversation_history:
            return ""
        
        # Limit to recent messages
        recent_history = self.conversation_history[-self.memory_length:]
        
        formatted_history = []
        for entry in recent_history:
            if entry["role"] == "user":
                formatted_history.append(f"User: {entry['content']}")
            elif entry["role"] == "assistant":
                formatted_history.append(f"Assistant: {entry['content']}")
        
        return "\n".join(formatted_history)
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a conversational prompt with history."""
        system_prompt = config.get("system_prompt", self._get_persona_prompt())
        context = config.get("context", "")
        query = config.get("query", "")
        
        # Build conversation history
        history_text = self._format_history()
        
        # Build background context section
        background_section = f"Background Information:\n{context}\n" if context else ""
        
        prompt = f"""{system_prompt}

{background_section}Current conversation:
{history_text}

User: {query}

Assistant:"""
        
        return prompt
    
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format context for conversational AI, prioritizing recent and relevant documents."""
        if not documents:
            return ""
        
        # Prioritize documents with higher relevance scores
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(sorted_docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Format document with metadata
            doc_text = content
            if metadata and self.config.get("include_metadata", False):
                source = metadata.get("source", f"Document {i+1}")
                doc_text = f"[Source: {source}]\n{content}"
            
            # Check length limit
            if current_length + len(doc_text) > self.config.get("max_context_length", 1500):
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n---\n".join(context_parts)


class CodeExplanationPrompter(BasePrompter):
    """Prompter specialized for generating code explanations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.context_separator = self.config.get("context_separator", "\n---\n")
    
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a prompt for code explanation."""
        system_prompt = config.get("system_prompt", "You are an expert in code explanation.")
        context = config.get("context", "")
        query = config.get("query", "")
        
        # Build the prompt
        prompt = f"""System: {system_prompt}

Context:
{context}

User: {query}

Assistant:"""
        
        return prompt
    
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format context for code-related queries, focusing on code snippets and documentation."""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Prioritize code snippets and relevant documentation
            if "code" in metadata.get("tags", []):
                doc_text = content
                
                # Add metadata as a comment in the code
                if metadata and self.config.get("include_metadata", False):
                    source = metadata.get("source", f"Document {i+1}")
                    doc_text = f"# Source: {source}\n{content}"
                
                # Check length limit
                if current_length + len(doc_text) > self.max_context_length:
                    break
                
                context_parts.append(doc_text)
                current_length += len(doc_text)
        
        return self.context_separator.join(context_parts)


class DebuggingPrompter(BasePrompter):
    """Prompter specialized for debugging assistance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.context_separator = self.config.get("context_separator", "\n---\n")
    
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """Build a prompt for debugging assistance."""
        system_prompt = config.get("system_prompt", "You are an expert debugging assistant.")
        context = config.get("context", "")
        query = config.get("query", "")
        
        # Build the prompt
        prompt = f"""System: {system_prompt}

Context:
{context}

User: {query}

Assistant:"""
        
        return prompt
    
    def format_context(self, documents: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Format context for debugging, focusing on error messages and relevant code sections."""
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Prioritize error messages and related code
            if "error" in metadata.get("tags", []):
                doc_text = content
                
                # Add metadata as a comment
                if metadata and self.config.get("include_metadata", False):
                    source = metadata.get("source", f"Document {i+1}")
                    doc_text = f"# Source: {source}\n{content}"
                
                # Check length limit
                if current_length + len(doc_text) > self.max_context_length:
                    break
                
                context_parts.append(doc_text)
                current_length += len(doc_text)
        
        return self.context_separator.join(context_parts)


def get_prompter(prompter_type: str, config: Dict[str, Any] = None) -> BasePrompter:
    """Factory function to get the appropriate prompter based on type."""
    prompter_classes = {
        "default": DefaultPrompter,
        "rag": RAGPrompter,
        "conversational": ConversationalPrompter,
        "code_explanation": CodeExplanationPrompter,
        "debugging": DebuggingPrompter,
    }
    
    prompter_class = prompter_classes.get(prompter_type, DefaultPrompter)
    return prompter_class(config)


# Legacy compatibility functions
def get_prompter_legacy(config: Dict[str, Any]) -> BasePrompter:
    """Legacy factory function for compatibility with existing code."""
    template_type = config.get("template", "default")
    return get_prompter(template_type, config)


# Backward compatibility with original base class
from rag_engine.core.base import BasePrompting

class DefaultPrompterLegacy(BasePrompting):
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.prompter = DefaultPrompter(config)
    
    def format(self, context, config):
        """Legacy method for compatibility with base class."""
        return self.prompter.build_prompt({
            "system_prompt": config.get("system_prompt", ""),
            "context": context,
            "query": config.get("query", "")
        })
