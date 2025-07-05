"""
Conversational RAG Integration Module.

Integrates the advanced conversational routing system with the existing RAG Engine
to provide human-like, context-aware responses that know when and how to use
RAG retrieval vs. other response strategies.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from rag_engine.core.conversational_routing import ConversationalRouter, ResponseStrategy
from rag_engine.core.base import BasePrompting

logger = logging.getLogger(__name__)


class ConversationalRAGPrompter(BasePrompting):
    """
    Advanced conversational prompter that uses multi-stage routing to determine
    the best response strategy for each query.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize conversational router
        self.router = ConversationalRouter(self.config)
        
        # Load routing templates
        self._load_routing_templates()
        
        # Track sessions
        self.sessions = {}
        
        # Integration settings
        self.enable_routing = self.config.get("enable_routing", True)
        self.fallback_to_simple = self.config.get("fallback_to_simple", True)
        
    def set_dependencies(self, llm=None, retriever=None, vectorstore=None):
        """Inject dependencies from the RAG system."""
        if llm:
            self.router.set_llm(llm)
        self.retriever = retriever
        self.vectorstore = vectorstore
        
    def build_prompt(self, config: Dict[str, Any]) -> str:
        """
        Build prompt using conversational routing system.
        
        This is the main entry point that determines whether to use
        RAG retrieval, contextual chat, or other response strategies.
        """
        query = config.get("query", "")
        session_id = config.get("session_id", "default")
        
        if not self.enable_routing:
            # Fall back to simple RAG prompt
            return self._build_simple_rag_prompt(config)
        
        try:
            # Route the query through the conversational system
            routing_result = self.router.route_query(query, session_id)
            
            # Handle different response strategies
            strategy = routing_result.get("strategy")
            
            if strategy == ResponseStrategy.RAG_RETRIEVAL.value:
                return self._build_rag_prompt_with_routing(config, routing_result)
            elif strategy in [ResponseStrategy.CONTEXTUAL_CHAT.value, 
                            ResponseStrategy.SIMPLE_RESPONSE.value,
                            ResponseStrategy.POLITE_REJECTION.value,
                            ResponseStrategy.CLARIFICATION_REQUEST.value]:
                # For non-RAG strategies, return the routed response directly
                return routing_result.get("response", "")
            else:
                # Fallback
                return self._build_simple_rag_prompt(config)
                
        except Exception as e:
            logger.error(f"Conversational routing failed: {e}")
            if self.fallback_to_simple:
                return self._build_simple_rag_prompt(config)
            else:
                raise
    
    def _build_rag_prompt_with_routing(self, config: Dict[str, Any], 
                                     routing_result: Dict[str, Any]) -> str:
        """Build RAG prompt enhanced with routing context."""
        query = config.get("query", "")
        context = config.get("context", "")
        system_prompt = config.get("system_prompt", 
            "You are a helpful AI assistant with access to documents.")
        
        # Extract routing insights
        classification = routing_result.get("classification", {})
        metadata = classification.get("metadata", {})
        entities = classification.get("entities", [])
        reasoning_chain = routing_result.get("reasoning_chain", [])
        
        # Build enhanced prompt with routing context
        enhanced_prompt = f"""{system_prompt}

QUERY ANALYSIS:
- Category: {classification.get('category', 'unknown')}
- Entities: {', '.join(entities) if entities else 'None identified'}
- User Intent: {classification.get('intent', 'Not specified')}
- Confidence: {routing_result.get('confidence', 0.0):.2f}

REASONING CHAIN:
{chr(10).join(f"- {step}" for step in reasoning_chain[-3:]) if reasoning_chain else "- Direct query processing"}

CONTEXT:
{context}

METADATA:
{metadata}

USER QUERY: {query}

INSTRUCTIONS:
Based on the analysis above, provide a comprehensive answer that:
1. Addresses the user's specific intent and expertise level
2. Uses the retrieved context effectively
3. Acknowledges the identified entities and their relevance
4. Maintains conversational flow if this is part of an ongoing discussion
5. Provides appropriate level of detail based on the query category

RESPONSE:"""
        
        return enhanced_prompt
    
    def _build_simple_rag_prompt(self, config: Dict[str, Any]) -> str:
        """Fallback to simple RAG prompt building."""
        system_prompt = config.get("system_prompt", 
            "You are a helpful assistant that answers questions based on the provided context.")
        context = config.get("context", "")
        query = config.get("query", "")
        
        if context:
            prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""{system_prompt}

Question: {query}

Answer:"""
        
        return prompt
    
    def format_context(self, documents: list, config: Dict[str, Any] = None) -> str:
        """Format retrieved documents for the prompt."""
        if not documents:
            return ""
        
        # Enhanced context formatting with routing awareness
        config = config or {}
        session_id = config.get("session_id", "default")
        
        # Get conversation context if available
        context_obj = self.router._get_context(session_id)
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0.0)
            
            # Format with citation numbers and metadata
            doc_str = f"[{i}] {content}"
            
            # Add source information if available
            if metadata.get("source"):
                doc_str += f"\n(Source: {metadata['source']})"
            
            # Add relevance score for debugging
            if config.get("include_scores", False):
                doc_str += f"\n(Relevance: {score:.3f})"
            
            formatted_docs.append(doc_str)
        
        return "\n\n".join(formatted_docs)
    
    def get_routing_insights(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Get routing analysis without generating a response."""
        try:
            routing_result = self.router.route_query(query, session_id)
            return {
                "category": routing_result.get("category"),
                "strategy": routing_result.get("strategy"), 
                "confidence": routing_result.get("confidence"),
                "topic_analysis": routing_result.get("topic_analysis"),
                "entities": routing_result.get("classification", {}).get("entities", []),
                "metadata": routing_result.get("metadata", {}),
                "reasoning": routing_result.get("reasoning_chain", [])
            }
        except Exception as e:
            logger.error(f"Failed to get routing insights: {e}")
            return {"error": str(e)}
    
    def _load_routing_templates(self):
        """Load routing templates from files."""
        template_dir = Path("templates/routing")
        
        # Update router with file-based templates if they exist
        if template_dir.exists():
            template_files = {
                "topic_analysis": "topic_analysis_template.txt",
                "query_classification": "query_classification_template.txt", 
                "rag_response": "rag_response_template.txt",
                "contextual_chat": "contextual_chat_template.txt",
                "polite_rejection": "polite_rejection_template.txt"
            }
            
            for prompt_type, filename in template_files.items():
                template_path = template_dir / filename
                if template_path.exists():
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            template_content = f.read()
                        # Store in router (would need to update router to accept this)
                        setattr(self.router, f"{prompt_type}_prompt", template_content)
                    except Exception as e:
                        logger.warning(f"Failed to load template {filename}: {e}")

    def format(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Format method required by BasePrompting interface.
        
        This delegates to the build_prompt method with context and config merged.
        """
        # Merge context and config for build_prompt
        merged_config = {**config, **context}
        return self.build_prompt(merged_config)
    

class ConversationalOrchestrator:
    """
    Orchestrator that integrates conversational routing with the RAG pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompter = ConversationalRAGPrompter(config.get("prompting", {}))
        
        # Will be injected by orchestration layer
        self.llm = None
        self.retriever = None
        self.vectorstore = None
        
    def set_components(self, llm=None, retriever=None, vectorstore=None):
        """Inject RAG components."""
        self.llm = llm
        self.retriever = retriever
        self.vectorstore = vectorstore
        
        # Pass to prompter
        self.prompter.set_dependencies(llm, retriever, vectorstore)
    
    def query(self, query: str, session_id: str = "default", **kwargs) -> Dict[str, Any]:
        """
        Process query with conversational routing.
        """
        try:
            # Get routing insights first
            insights = self.prompter.get_routing_insights(query, session_id)
            strategy = insights.get("strategy")
            
            # Handle based on strategy
            if strategy == "rag_retrieval":
                return self._handle_rag_query(query, session_id, insights, **kwargs)
            else:
                return self._handle_non_rag_query(query, session_id, insights, **kwargs)
                
        except Exception as e:
            logger.error(f"Conversational orchestrator error: {e}")
            return {
                "answer": "I apologize, but I'm having trouble processing your request.",
                "error": str(e),
                "strategy": "error"
            }
    
    def _handle_rag_query(self, query: str, session_id: str, 
                         insights: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Handle queries that require RAG retrieval."""
        # Retrieve relevant documents
        retrieved_docs = []
        if self.retriever and self.vectorstore:
            try:
                retrieved_docs = self.retriever.retrieve(
                    self.vectorstore, query, kwargs.get("retrieval_config", {})
                )
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
        
        # Format context
        context = self.prompter.format_context(retrieved_docs, {
            "session_id": session_id,
            "include_scores": kwargs.get("include_scores", False)
        })
        
        # Build prompt with routing context
        prompt_config = {
            "query": query,
            "context": context,
            "session_id": session_id,
            "system_prompt": self.config.get("prompting", {}).get("system_prompt", "")
        }
        
        prompt = self.prompter.build_prompt(prompt_config)
        
        # Generate response
        if self.llm:
            response = self.llm.generate(prompt, self.config.get("llm", {}))
        else:
            response = "LLM not available"
        
        return {
            "answer": response,
            "strategy": "rag_retrieval",
            "sources": [doc.get("metadata", {}) for doc in retrieved_docs],
            "retrieved_docs": len(retrieved_docs),
            "insights": insights,
            "confidence": insights.get("confidence", 0.0)
        }
    
    def _handle_non_rag_query(self, query: str, session_id: str,
                            insights: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Handle queries that don't require RAG retrieval."""
        # Use the routing system's response directly
        try:
            routing_result = self.prompter.router.route_query(query, session_id)
            return {
                "answer": routing_result.get("response", ""),
                "strategy": routing_result.get("strategy"),
                "category": routing_result.get("category"),
                "insights": insights,
                "confidence": routing_result.get("confidence", 0.0),
                "reasoning": routing_result.get("reasoning_chain", [])
            }
        except Exception as e:
            return {
                "answer": "I'm having trouble generating a response right now.",
                "error": str(e),
                "strategy": "error"
            }


# Factory function for creating conversational RAG prompter
def create_conversational_rag_prompter(config: Dict[str, Any]) -> ConversationalRAGPrompter:
    """Create conversational RAG prompter."""
    return ConversationalRAGPrompter(config)


# Factory function for creating conversational orchestrator  
def create_conversational_orchestrator(config: Dict[str, Any]) -> ConversationalOrchestrator:
    """Create conversational orchestrator."""
    return ConversationalOrchestrator(config)
