"""
Advanced Conversational Routing System for RAG Engine.

This module implements a sophisticated multi-stage LLM routing system that:
1. Detects topic, checks if broad/ambiguous
2. Classifies query and extracts metadata  
3. Routes to appropriate response strategy (RAG, simple chat, rejection, etc.)
4. Maintains conversation context and chain-of-thought reasoning

This addresses the challenge of creating human-like conversational agents
that know when to use RAG vs. other response strategies.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """Categories for query classification."""
    RAG_FACTUAL = "rag_factual"           # Requires RAG lookup
    RAG_ANALYTICAL = "rag_analytical"     # Requires RAG + analysis
    GREETING = "greeting"                 # Simple greeting
    GOODBYE = "goodbye"                   # Farewell
    GRATITUDE = "gratitude"              # Thank you, etc.
    CLARIFICATION = "clarification"       # Asking for clarification
    FOLLOW_UP = "follow_up"              # Following up on previous
    OUT_OF_CONTEXT = "out_of_context"    # Outside domain
    SMALL_TALK = "small_talk"            # Casual conversation
    HELP_REQUEST = "help_request"        # Asking for help/guidance


class ResponseStrategy(Enum):
    """Response strategies based on query classification."""
    RAG_RETRIEVAL = "rag_retrieval"      # Full RAG pipeline
    CONTEXTUAL_CHAT = "contextual_chat"  # Chat with conversation context
    SIMPLE_RESPONSE = "simple_response"  # Direct simple response
    POLITE_REJECTION = "polite_rejection" # Politely decline/redirect
    CLARIFICATION_REQUEST = "clarification_request" # Ask for clarification


@dataclass
class TopicAnalysis:
    """Results from topic detection stage."""
    topic: str
    is_broad: bool
    is_ambiguous: bool
    confidence: float
    reasoning: str


@dataclass
class QueryClassification:
    """Results from query classification stage."""
    category: QueryCategory
    metadata: Dict[str, Any]
    entities: List[str]
    intent: str
    confidence: float
    reasoning: str


@dataclass
class ConversationContext:
    """Maintains conversation state and context."""
    history: List[Dict[str, Any]]
    accumulated_metadata: Dict[str, Any]
    chain_of_thought: List[str]
    user_profile: Dict[str, Any]
    session_id: str
    
    def add_exchange(self, user_query: str, assistant_response: str, 
                    metadata: Dict[str, Any] = None):
        """Add a conversation exchange."""
        self.history.append({
            "user": user_query,
            "assistant": assistant_response,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        })
        
        # Update accumulated metadata
        if metadata:
            self.accumulated_metadata.update(metadata)
    
    def add_reasoning_step(self, step: str):
        """Add a reasoning step to chain of thought."""
        self.chain_of_thought.append(step)
    
    def get_context_summary(self, max_exchanges: int = 5) -> str:
        """Get a summary of recent conversation context."""
        recent_history = self.history[-max_exchanges:] if self.history else []
        
        context_parts = []
        if recent_history:
            context_parts.append("Recent conversation:")
            for exchange in recent_history:
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"Assistant: {exchange['assistant']}")
        
        if self.accumulated_metadata:
            context_parts.append(f"Context metadata: {json.dumps(self.accumulated_metadata, indent=2)}")
        
        return "\n".join(context_parts)
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class ConversationalRouter:
    """Advanced conversational routing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None  # Will be injected
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Load routing prompts
        self.topic_analysis_prompt = self._load_prompt("topic_analysis")
        self.classification_prompt = self._load_prompt("query_classification") 
        self.rag_prompt = self._load_prompt("rag_response")
        self.chat_prompt = self._load_prompt("contextual_chat")
        self.rejection_prompt = self._load_prompt("polite_rejection")
    
    def set_llm(self, llm):
        """Inject LLM dependency."""
        self.llm = llm
    
    def route_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Main routing function that processes a query through the full pipeline.
        
        Returns complete response with reasoning chain.
        """
        try:
            # Get or create conversation context
            context = self._get_context(session_id)
            context.add_reasoning_step(f"Processing query: {query}")
            
            # Stage 1: Topic Analysis
            topic_analysis = self._analyze_topic(query, context)
            context.add_reasoning_step(f"Topic analysis: {topic_analysis.reasoning}")
            
            # Stage 2: Query Classification
            classification = self._classify_query(query, topic_analysis, context)
            context.add_reasoning_step(f"Classification: {classification.reasoning}")
            
            # Stage 3: Route to appropriate strategy
            strategy = self._determine_strategy(classification, topic_analysis)
            context.add_reasoning_step(f"Selected strategy: {strategy.value}")
            
            # Stage 4: Generate response
            response = self._generate_response(query, strategy, classification, context)
            
            # Update conversation context
            context.add_exchange(
                user_query=query,
                assistant_response=response["content"],
                metadata=classification.metadata
            )
            
            return {
                "response": response["content"],
                "strategy": strategy.value,
                "category": classification.category.value,
                "topic_analysis": asdict(topic_analysis),
                "classification": asdict(classification),
                "reasoning_chain": context.chain_of_thought[-10:],  # Last 10 steps
                "confidence": classification.confidence,
                "metadata": classification.metadata
            }
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            return self._fallback_response(query, str(e))
    
    def _analyze_topic(self, query: str, context: ConversationContext) -> TopicAnalysis:
        """Stage 1: Analyze topic, detect if broad/ambiguous."""
        prompt = self.topic_analysis_prompt.format(
            query=query,
            context=context.get_context_summary()
        )
        
        response = self._call_llm(prompt, "topic_analysis")
        
        # Parse LLM response (expected JSON format)
        try:
            result = json.loads(response)
            return TopicAnalysis(
                topic=result.get("topic", "unknown"),
                is_broad=result.get("is_broad", False),
                is_ambiguous=result.get("is_ambiguous", False),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError:
            # Fallback parsing
            return TopicAnalysis(
                topic="general",
                is_broad=True,
                is_ambiguous=True,
                confidence=0.3,
                reasoning="Failed to parse topic analysis"
            )
    
    def _classify_query(self, query: str, topic_analysis: TopicAnalysis, 
                       context: ConversationContext) -> QueryClassification:
        """Stage 2: Classify query and extract metadata."""
        prompt = self.classification_prompt.format(
            query=query,
            topic=topic_analysis.topic,
            is_broad=topic_analysis.is_broad,
            is_ambiguous=topic_analysis.is_ambiguous,
            context=context.get_context_summary()
        )
        
        response = self._call_llm(prompt, "classification")
        
        try:
            result = json.loads(response)
            
            # Map category string to enum
            category_str = result.get("category", "out_of_context")
            try:
                category = QueryCategory(category_str)
            except ValueError:
                category = QueryCategory.OUT_OF_CONTEXT
            
            return QueryClassification(
                category=category,
                metadata=result.get("metadata", {}),
                entities=result.get("entities", []),
                intent=result.get("intent", ""),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError:
            return QueryClassification(
                category=QueryCategory.OUT_OF_CONTEXT,
                metadata={},
                entities=[],
                intent="unknown",
                confidence=0.3,
                reasoning="Failed to parse classification"
            )
    
    def _determine_strategy(self, classification: QueryClassification, 
                          topic_analysis: TopicAnalysis) -> ResponseStrategy:
        """Determine response strategy based on classification."""
        category = classification.category
        
        # Map categories to strategies
        strategy_map = {
            QueryCategory.RAG_FACTUAL: ResponseStrategy.RAG_RETRIEVAL,
            QueryCategory.RAG_ANALYTICAL: ResponseStrategy.RAG_RETRIEVAL,
            QueryCategory.GREETING: ResponseStrategy.SIMPLE_RESPONSE,
            QueryCategory.GOODBYE: ResponseStrategy.SIMPLE_RESPONSE,
            QueryCategory.GRATITUDE: ResponseStrategy.SIMPLE_RESPONSE,
            QueryCategory.CLARIFICATION: ResponseStrategy.CLARIFICATION_REQUEST,
            QueryCategory.FOLLOW_UP: ResponseStrategy.CONTEXTUAL_CHAT,
            QueryCategory.OUT_OF_CONTEXT: ResponseStrategy.POLITE_REJECTION,
            QueryCategory.SMALL_TALK: ResponseStrategy.CONTEXTUAL_CHAT,
            QueryCategory.HELP_REQUEST: ResponseStrategy.CONTEXTUAL_CHAT,
        }
        
        strategy = strategy_map.get(category, ResponseStrategy.POLITE_REJECTION)
        
        # Adjust strategy based on topic analysis
        if topic_analysis.is_ambiguous and strategy == ResponseStrategy.RAG_RETRIEVAL:
            strategy = ResponseStrategy.CLARIFICATION_REQUEST
        
        return strategy
    
    def _generate_response(self, query: str, strategy: ResponseStrategy,
                          classification: QueryClassification,
                          context: ConversationContext) -> Dict[str, Any]:
        """Stage 3: Generate response based on strategy."""
        
        if strategy == ResponseStrategy.RAG_RETRIEVAL:
            return self._generate_rag_response(query, classification, context)
        
        elif strategy == ResponseStrategy.CONTEXTUAL_CHAT:
            return self._generate_contextual_response(query, classification, context)
        
        elif strategy == ResponseStrategy.SIMPLE_RESPONSE:
            return self._generate_simple_response(query, classification)
        
        elif strategy == ResponseStrategy.POLITE_REJECTION:
            return self._generate_rejection_response(query, classification, context)
        
        elif strategy == ResponseStrategy.CLARIFICATION_REQUEST:
            return self._generate_clarification_response(query, classification, context)
        
        else:
            return {"content": "I apologize, but I'm not sure how to respond to that."}
    
    def _generate_rag_response(self, query: str, classification: QueryClassification,
                              context: ConversationContext) -> Dict[str, Any]:
        """Generate response using RAG pipeline."""
        # This would integrate with the main RAG system
        prompt = self.rag_prompt.format(
            query=query,
            metadata=json.dumps(classification.metadata),
            context=context.get_context_summary(),
            entities=", ".join(classification.entities)
        )
        
        response = self._call_llm(prompt, "rag_response")
        return {"content": response, "type": "rag"}
    
    def _generate_contextual_response(self, query: str, classification: QueryClassification,
                                    context: ConversationContext) -> Dict[str, Any]:
        """Generate contextual chat response."""
        prompt = self.chat_prompt.format(
            query=query,
            context=context.get_context_summary(),
            metadata=json.dumps(classification.metadata),
            reasoning_chain="\n".join(context.chain_of_thought[-3:])
        )
        
        response = self._call_llm(prompt, "contextual_chat")
        return {"content": response, "type": "contextual"}
    
    def _generate_simple_response(self, query: str, 
                                classification: QueryClassification) -> Dict[str, Any]:
        """Generate simple response for greetings, etc."""
        category = classification.category
        
        simple_responses = {
            QueryCategory.GREETING: "Hello! I'm here to help you with any questions about our documents. How can I assist you today?",
            QueryCategory.GOODBYE: "Goodbye! Feel free to come back if you have more questions.",
            QueryCategory.GRATITUDE: "You're welcome! I'm glad I could help. Is there anything else you'd like to know?"
        }
        
        response = simple_responses.get(category, "Thank you for your message.")
        return {"content": response, "type": "simple"}
    
    def _generate_rejection_response(self, query: str, classification: QueryClassification,
                                   context: ConversationContext) -> Dict[str, Any]:
        """Generate polite rejection for out-of-context queries."""
        prompt = self.rejection_prompt.format(
            query=query,
            context=context.get_context_summary(),
            reasoning=classification.reasoning
        )
        
        response = self._call_llm(prompt, "polite_rejection")
        return {"content": response, "type": "rejection"}
    
    def _generate_clarification_response(self, query: str, classification: QueryClassification,
                                       context: ConversationContext) -> Dict[str, Any]:
        """Generate clarification request."""
        clarifications = [
            "I'd be happy to help! Could you provide a bit more detail about what specifically you're looking for?",
            "That's an interesting question! Could you clarify what aspect you'd like me to focus on?",
            "I want to make sure I give you the most helpful answer. Could you be more specific about your question?"
        ]
        
        # Could use LLM for more sophisticated clarification
        response = clarifications[0]  # Simple fallback
        return {"content": response, "type": "clarification"}
    
    def _get_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context."""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = ConversationContext(
                history=[],
                accumulated_metadata={},
                chain_of_thought=[],
                user_profile={},
                session_id=session_id
            )
        return self.conversation_contexts[session_id]
    
    def _call_llm(self, prompt: str, call_type: str) -> str:
        """Call LLM with prompt."""
        if not self.llm:
            raise RuntimeError("LLM not configured")
        
        try:
            # Configure LLM call based on type
            config = self._get_llm_config(call_type)
            response = self.llm.generate(prompt, config)
            return response
        except Exception as e:
            logger.error(f"LLM call failed for {call_type}: {e}")
            return f"Error: {e}"
    
    def _get_llm_config(self, call_type: str) -> Dict[str, Any]:
        """Get LLM configuration for different call types."""
        base_config = self.config.get("llm", {})
        
        # Adjust temperature and parameters based on call type
        configs = {
            "topic_analysis": {"temperature": 0.1, "max_tokens": 300},
            "classification": {"temperature": 0.1, "max_tokens": 400},
            "rag_response": {"temperature": 0.3, "max_tokens": 800},
            "contextual_chat": {"temperature": 0.7, "max_tokens": 600},
            "polite_rejection": {"temperature": 0.5, "max_tokens": 300}
        }
        
        call_config = configs.get(call_type, {})
        return {**base_config, **call_config}
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load prompt template for specific routing stage."""
        # These would be loaded from files or configured
        prompts = {
            "topic_analysis": self._get_topic_analysis_prompt(),
            "query_classification": self._get_classification_prompt(),
            "rag_response": self._get_rag_prompt(),
            "contextual_chat": self._get_chat_prompt(),
            "polite_rejection": self._get_rejection_prompt()
        }
        
        return prompts.get(prompt_type, "")
    
    def _get_topic_analysis_prompt(self) -> str:
        """Topic analysis prompt template."""
        return """Analyze the user's query to understand the topic and determine if it's broad or ambiguous.

Query: {query}

Previous context:
{context}

Please respond with JSON in this format:
{{
    "topic": "main topic of the query",
    "is_broad": true/false,
    "is_ambiguous": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your analysis"
}}

Consider:
- Is this query about a specific topic or very general?
- Could this query be interpreted in multiple ways?
- Is there enough information to provide a focused answer?"""
    
    def _get_classification_prompt(self) -> str:
        """Query classification prompt template."""
        return """Classify the user's query and extract relevant metadata.

Query: {query}
Topic: {topic}
Is Broad: {is_broad}
Is Ambiguous: {is_ambiguous}

Previous context:
{context}

Classify into one of these categories:
- rag_factual: Needs factual information lookup
- rag_analytical: Needs analysis of retrieved information  
- greeting: Hello, hi, etc.
- goodbye: Bye, farewell, etc.
- gratitude: Thank you, thanks, etc.
- clarification: Asking for clarification
- follow_up: Following up on previous conversation
- out_of_context: Outside the domain/scope
- small_talk: Casual conversation
- help_request: Asking for help or guidance

Respond with JSON:
{{
    "category": "category_name",
    "metadata": {{"key": "extracted metadata"}},
    "entities": ["named entities"],
    "intent": "user's intent",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""
    
    def _get_rag_prompt(self) -> str:
        """RAG response prompt template."""
        return """Generate a helpful response based on retrieved information.

User Query: {query}
Metadata: {metadata}
Entities: {entities}

Conversation Context:
{context}

Instructions:
- Use the retrieved documents to answer the user's question
- Be specific and cite sources when possible
- Maintain conversation flow and context
- If information is incomplete, acknowledge limitations"""
    
    def _get_chat_prompt(self) -> str:
        """Contextual chat prompt template."""
        return """Generate a conversational response that maintains context.

User Query: {query}
Metadata: {metadata}

Conversation Context:
{context}

Recent Reasoning:
{reasoning_chain}

Instructions:
- Respond naturally and helpfully
- Use conversation context appropriately
- Be engaging and maintain rapport
- Stay helpful and professional"""
    
    def _get_rejection_prompt(self) -> str:
        """Polite rejection prompt template."""
        return """Generate a polite response for queries outside your domain.

User Query: {query}
Classification Reasoning: {reasoning}

Conversation Context:
{context}

Instructions:
- Be polite and respectful
- Explain limitations diplomatically
- Suggest alternatives if possible
- Maintain helpful tone
- Don't be abrupt or dismissive"""
    
    def _fallback_response(self, query: str, error: str) -> Dict[str, Any]:
        """Fallback response when routing fails."""
        return {
            "response": "I apologize, but I'm having trouble processing your request right now. Could you please try rephrasing your question?",
            "strategy": "fallback",
            "category": "error", 
            "error": error,
            "confidence": 0.0
        }


# Factory function for creating conversational router
def create_conversational_router(config: Dict[str, Any]) -> ConversationalRouter:
    """Create and configure a conversational router."""
    return ConversationalRouter(config)
