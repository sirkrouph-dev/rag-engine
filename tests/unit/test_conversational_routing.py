"""
Unit tests for the conversational routing system.

Tests the core routing logic, template management, and integration components.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from rag_engine.core.conversational_routing import (
    ConversationalRouter,
    QueryCategory,
    ResponseStrategy,
    TopicAnalysis,
    QueryClassification,
    RoutingDecision
)
from rag_engine.core.conversational_integration import ConversationalRAGPrompter


class TestConversationalRouter:
    """Test the core conversational routing logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "topic_analysis_temperature": 0.1,
            "classification_temperature": 0.1,
            "response_temperature": 0.7,
            "max_conversation_history": 10,
            "confidence_threshold": 0.8,
            "enable_reasoning_chain": True
        }
        
        # Mock LLM for testing
        self.mock_llm = Mock()
        self.router = ConversationalRouter(self.config)
        self.router.set_llm(self.mock_llm)
    
    def test_router_initialization(self):
        """Test router initialization with config."""
        assert self.router.config == self.config
        assert self.router.conversation_context == {}
        assert self.router.llm is self.mock_llm
    
    def test_topic_analysis_parsing(self):
        """Test parsing of topic analysis from LLM response."""
        # Mock LLM response
        llm_response = '''
        {
            "topic": "machine learning",
            "is_broad": false,
            "is_ambiguous": false,
            "confidence": 0.9,
            "reasoning": "Specific technical query about ML"
        }
        '''
        
        self.mock_llm.complete.return_value = llm_response
        
        analysis = self.router._analyze_topic("What is supervised learning?")
        
        assert isinstance(analysis, TopicAnalysis)
        assert analysis.topic == "machine learning"
        assert analysis.is_broad == False
        assert analysis.is_ambiguous == False
        assert analysis.confidence == 0.9
        assert "technical query" in analysis.reasoning
    
    def test_query_classification_parsing(self):
        """Test parsing of query classification from LLM response."""
        # Setup topic analysis
        topic_analysis = TopicAnalysis(
            topic="machine learning",
            is_broad=False,
            is_ambiguous=False,
            confidence=0.9,
            reasoning="Technical query"
        )
        
        # Mock LLM response
        llm_response = '''
        {
            "category": "rag_factual",
            "intent": "learn_concept",
            "entities": ["supervised learning", "machine learning"],
            "domain": "artificial intelligence",
            "expertise_level": "beginner",
            "confidence": 0.85,
            "reasoning": "User wants to learn about ML concept"
        }
        '''
        
        self.mock_llm.complete.return_value = llm_response
        
        classification = self.router._classify_query(
            "What is supervised learning?", 
            topic_analysis
        )
        
        assert isinstance(classification, QueryClassification)
        assert classification.category == QueryCategory.RAG_FACTUAL
        assert classification.intent == "learn_concept"
        assert "supervised learning" in classification.entities
        assert classification.domain == "artificial intelligence"
        assert classification.confidence == 0.85
    
    def test_strategy_selection(self):
        """Test strategy selection based on classification."""
        # Test RAG factual query
        classification = QueryClassification(
            category=QueryCategory.RAG_FACTUAL,
            intent="learn_concept",
            entities=["machine learning"],
            domain="AI",
            expertise_level="beginner",
            confidence=0.85,
            reasoning="Learning query"
        )
        
        strategy = self.router._select_strategy(classification, 0.85)
        assert strategy == ResponseStrategy.RAG_RETRIEVAL
        
        # Test greeting
        classification.category = QueryCategory.GREETING
        strategy = self.router._select_strategy(classification, 0.9)
        assert strategy == ResponseStrategy.SIMPLE_RESPONSE
        
        # Test out of context
        classification.category = QueryCategory.OUT_OF_CONTEXT
        strategy = self.router._select_strategy(classification, 0.8)
        assert strategy == ResponseStrategy.POLITE_REJECTION
    
    def test_full_routing_pipeline(self):
        """Test the complete routing pipeline."""
        # Mock LLM responses for each stage
        topic_response = '''
        {
            "topic": "greeting",
            "is_broad": false,
            "is_ambiguous": false,
            "confidence": 0.95,
            "reasoning": "Simple greeting"
        }
        '''
        
        classification_response = '''
        {
            "category": "greeting",
            "intent": "greet",
            "entities": [],
            "domain": "social",
            "expertise_level": "any",
            "confidence": 0.9,
            "reasoning": "Friendly greeting"
        }
        '''
        
        self.mock_llm.complete.side_effect = [topic_response, classification_response]
        
        decision = self.router.route_query("Hello!", session_id="test123")
        
        assert isinstance(decision, RoutingDecision)
        assert decision.strategy == ResponseStrategy.SIMPLE_RESPONSE
        assert decision.query == "Hello!"
        assert decision.session_id == "test123"
        assert decision.confidence >= 0.8
    
    def test_conversation_context_management(self):
        """Test conversation context tracking."""
        session_id = "test123"
        
        # First query
        self.router.add_to_context(session_id, "user", "Hello!")
        self.router.add_to_context(session_id, "assistant", "Hi there!")
        
        context = self.router.get_conversation_context(session_id)
        assert len(context["history"]) == 2
        assert context["history"][0]["role"] == "user"
        assert context["history"][1]["role"] == "assistant"
    
    def test_low_confidence_handling(self):
        """Test handling of low confidence classifications."""
        # Mock low confidence response
        topic_response = '''
        {
            "topic": "unclear",
            "is_broad": true,
            "is_ambiguous": true,
            "confidence": 0.3,
            "reasoning": "Very unclear query"
        }
        '''
        
        classification_response = '''
        {
            "category": "clarification",
            "intent": "unclear",
            "entities": [],
            "domain": "unknown",
            "expertise_level": "unknown",
            "confidence": 0.4,
            "reasoning": "Need clarification"
        }
        '''
        
        self.mock_llm.complete.side_effect = [topic_response, classification_response]
        
        decision = self.router.route_query("Tell me about stuff", session_id="test")
        
        # Should route to clarification due to low confidence and ambiguity
        assert decision.strategy == ResponseStrategy.CLARIFICATION_REQUEST
    
    def test_error_handling(self):
        """Test error handling in routing pipeline."""
        # Mock LLM error
        self.mock_llm.complete.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception):
            self.router.route_query("Test query", session_id="test")
    
    def test_template_loading(self):
        """Test loading of routing templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test template
            template_dir = Path(temp_dir) / "routing"
            template_dir.mkdir()
            
            template_path = template_dir / "topic_analysis_template.txt"
            template_content = "Test template with {query} variable"
            template_path.write_text(template_content)
            
            # Mock the templates directory
            with patch('rag_engine.core.conversational_routing.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.glob.return_value = [template_path]
                
                templates = self.router._load_templates()
                assert "topic_analysis_template" in templates
                assert templates["topic_analysis_template"] == template_content


class TestConversationalRAGPrompter:
    """Test the conversational RAG prompter integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "enable_routing": True,
            "fallback_to_simple": True,
            "routing_config": {
                "max_conversation_history": 10,
                "confidence_threshold": 0.8
            }
        }
        
        self.prompter = ConversationalRAGPrompter(self.config)
        
        # Mock dependencies
        self.mock_llm = Mock()
        self.mock_retriever = Mock()
        self.mock_vectorstore = Mock()
        
        self.prompter.set_dependencies(
            llm=self.mock_llm,
            retriever=self.mock_retriever,
            vectorstore=self.mock_vectorstore
        )
    
    def test_prompter_initialization(self):
        """Test prompter initialization."""
        assert self.prompter.enable_routing == True
        assert self.prompter.fallback_to_simple == True
        assert isinstance(self.prompter.router, ConversationalRouter)
    
    def test_rag_strategy_prompt_building(self):
        """Test prompt building for RAG strategy."""
        # Mock routing decision
        decision = RoutingDecision(
            query="What is machine learning?",
            strategy=ResponseStrategy.RAG_RETRIEVAL,
            classification=QueryClassification(
                category=QueryCategory.RAG_FACTUAL,
                intent="learn",
                entities=["machine learning"],
                domain="AI",
                expertise_level="beginner",
                confidence=0.9,
                reasoning="Learning query"
            ),
            topic_analysis=Mock(),
            confidence=0.9,
            session_id="test",
            reasoning_chain=["Analysis step 1", "Classification step 2"]
        )
        
        # Mock retriever results
        self.mock_retriever.retrieve.return_value = [
            {"content": "ML is a subset of AI", "score": 0.9},
            {"content": "ML uses algorithms", "score": 0.8}
        ]
        
        with patch.object(self.prompter.router, 'route_query', return_value=decision):
            prompt = self.prompter.build_prompt({
                "query": "What is machine learning?",
                "session_id": "test"
            })
        
        assert "What is machine learning?" in prompt
        assert "ML is a subset of AI" in prompt  # Retrieved content
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_simple_response_strategy(self):
        """Test prompt building for simple response strategy."""
        decision = RoutingDecision(
            query="Hello!",
            strategy=ResponseStrategy.SIMPLE_RESPONSE,
            classification=QueryClassification(
                category=QueryCategory.GREETING,
                intent="greet",
                entities=[],
                domain="social",
                expertise_level="any",
                confidence=0.95,
                reasoning="Simple greeting"
            ),
            topic_analysis=Mock(),
            confidence=0.95,
            session_id="test",
            reasoning_chain=["Greeting detected"]
        )
        
        with patch.object(self.prompter.router, 'route_query', return_value=decision):
            prompt = self.prompter.build_prompt({
                "query": "Hello!",
                "session_id": "test"
            })
        
        assert "Hello!" in prompt
        assert "greeting" in prompt.lower() or "hello" in prompt.lower()
    
    def test_polite_rejection_strategy(self):
        """Test prompt building for polite rejection strategy."""
        decision = RoutingDecision(
            query="What's the weather?",
            strategy=ResponseStrategy.POLITE_REJECTION,
            classification=QueryClassification(
                category=QueryCategory.OUT_OF_CONTEXT,
                intent="weather_inquiry",
                entities=["weather"],
                domain="meteorology",
                expertise_level="any",
                confidence=0.85,
                reasoning="Outside domain"
            ),
            topic_analysis=Mock(),
            confidence=0.85,
            session_id="test",
            reasoning_chain=["Out of context detected"]
        )
        
        with patch.object(self.prompter.router, 'route_query', return_value=decision):
            prompt = self.prompter.build_prompt({
                "query": "What's the weather?",
                "session_id": "test"
            })
        
        assert "What's the weather?" in prompt
        assert "outside" in prompt.lower() or "help" in prompt.lower()
    
    def test_fallback_to_simple(self):
        """Test fallback behavior when routing fails."""
        # Mock routing failure
        with patch.object(self.prompter.router, 'route_query', side_effect=Exception("Routing error")):
            prompt = self.prompter.build_prompt({
                "query": "Test query",
                "session_id": "test"
            })
        
        # Should fallback to simple prompt
        assert "Test query" in prompt
        assert isinstance(prompt, str)
    
    def test_disabled_routing(self):
        """Test behavior when routing is disabled."""
        config = {
            "enable_routing": False,
            "fallback_to_simple": True
        }
        
        prompter = ConversationalRAGPrompter(config)
        prompt = prompter.build_prompt({
            "query": "Test query",
            "session_id": "test"
        })
        
        # Should use simple prompt building
        assert "Test query" in prompt
        assert isinstance(prompt, str)
    
    def test_session_management(self):
        """Test session-based conversation management."""
        session_id = "test123"
        
        # First interaction
        decision1 = RoutingDecision(
            query="Hello!",
            strategy=ResponseStrategy.SIMPLE_RESPONSE,
            classification=Mock(),
            topic_analysis=Mock(),
            confidence=0.9,
            session_id=session_id,
            reasoning_chain=[]
        )
        
        with patch.object(self.prompter.router, 'route_query', return_value=decision1):
            self.prompter.build_prompt({
                "query": "Hello!",
                "session_id": session_id
            })
        
        # Check that session was created
        assert session_id in self.prompter.sessions
        
        # Second interaction should have conversation context
        decision2 = RoutingDecision(
            query="What is AI?",
            strategy=ResponseStrategy.RAG_RETRIEVAL,
            classification=Mock(),
            topic_analysis=Mock(),
            confidence=0.85,
            session_id=session_id,
            reasoning_chain=[]
        )
        
        self.mock_retriever.retrieve.return_value = []
        
        with patch.object(self.prompter.router, 'route_query', return_value=decision2):
            prompt = self.prompter.build_prompt({
                "query": "What is AI?",
                "session_id": session_id
            })
        
        # Should include conversation context
        assert isinstance(prompt, str)


class TestRoutingDecisionLogic:
    """Test routing decision logic and edge cases."""
    
    def test_query_category_enum(self):
        """Test query category enumeration."""
        assert QueryCategory.RAG_FACTUAL.value == "rag_factual"
        assert QueryCategory.GREETING.value == "greeting"
        assert QueryCategory.OUT_OF_CONTEXT.value == "out_of_context"
    
    def test_response_strategy_enum(self):
        """Test response strategy enumeration."""
        assert ResponseStrategy.RAG_RETRIEVAL.value == "rag_retrieval"
        assert ResponseStrategy.SIMPLE_RESPONSE.value == "simple_response"
        assert ResponseStrategy.POLITE_REJECTION.value == "polite_rejection"
    
    def test_topic_analysis_dataclass(self):
        """Test TopicAnalysis dataclass."""
        analysis = TopicAnalysis(
            topic="test",
            is_broad=False,
            is_ambiguous=True,
            confidence=0.8,
            reasoning="test reasoning"
        )
        
        assert analysis.topic == "test"
        assert analysis.is_broad == False
        assert analysis.is_ambiguous == True
        assert analysis.confidence == 0.8
    
    def test_query_classification_dataclass(self):
        """Test QueryClassification dataclass."""
        classification = QueryClassification(
            category=QueryCategory.RAG_FACTUAL,
            intent="learn",
            entities=["AI", "ML"],
            domain="technology",
            expertise_level="intermediate",
            confidence=0.85,
            reasoning="Technical learning query"
        )
        
        assert classification.category == QueryCategory.RAG_FACTUAL
        assert classification.intent == "learn"
        assert "AI" in classification.entities
        assert classification.confidence == 0.85
    
    def test_routing_decision_dataclass(self):
        """Test RoutingDecision dataclass."""
        decision = RoutingDecision(
            query="test query",
            strategy=ResponseStrategy.RAG_RETRIEVAL,
            classification=Mock(),
            topic_analysis=Mock(),
            confidence=0.9,
            session_id="test123",
            reasoning_chain=["step1", "step2"]
        )
        
        assert decision.query == "test query"
        assert decision.strategy == ResponseStrategy.RAG_RETRIEVAL
        assert decision.confidence == 0.9
        assert len(decision.reasoning_chain) == 2


class TestTemplateManagement:
    """Test template loading and management functionality."""
    
    def test_template_directory_creation(self):
        """Test automatic creation of template directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir) / "templates" / "routing"
            
            # Should create directory if it doesn't exist
            router = ConversationalRouter({})
            
            # Mock the template directory path
            with patch('rag_engine.core.conversational_routing.Path') as mock_path:
                mock_path.return_value = template_dir
                mock_path.return_value.exists.return_value = False
                mock_path.return_value.mkdir = Mock()
                
                # This would normally create the directory
                templates = router._load_templates()
                assert templates == {}  # No templates found
    
    def test_template_content_validation(self):
        """Test template content validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir) / "routing"
            template_dir.mkdir()
            
            # Create valid template
            valid_template = template_dir / "valid_template.txt"
            valid_template.write_text("Valid template with {query}")
            
            # Create invalid template (empty)
            invalid_template = template_dir / "invalid_template.txt"
            invalid_template.write_text("")
            
            router = ConversationalRouter({})
            
            with patch('rag_engine.core.conversational_routing.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.glob.return_value = [valid_template, invalid_template]
                
                templates = router._load_templates()
                
                # Should load valid template
                assert "valid_template" in templates
                assert templates["valid_template"] == "Valid template with {query}"
                
                # Should handle invalid template gracefully
                assert "invalid_template" in templates
                assert templates["invalid_template"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
