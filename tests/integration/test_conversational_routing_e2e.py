"""
End-to-End tests for the conversational routing system.

Tests the complete user workflow including UI interactions,
API calls, and system integration.
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the components we need to test
from rag_engine.core.conversational_routing import (
    ConversationalRouter,
    QueryCategory,
    ResponseStrategy,
    TopicAnalysis,
    QueryClassification,
    ConversationContext
)
from rag_engine.core.conversational_integration import ConversationalRAGPrompter


class TestConversationalRoutingE2E:
    """End-to-end tests for conversational routing workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "prompting": {
                "routing_config": {
                    "llm_config": {"type": "mock"},
                    "templates_dir": "templates/routing",
                    "max_conversation_history": 20
                }
            }
        }
        
        self.mock_templates = {
            "topic_analysis": "Mock topic analysis template",
            "query_classification": "Mock classification template",
            "rag_response": "Mock RAG template",
            "contextual_chat": "Mock chat template",
            "polite_rejection": "Mock rejection template"
        }

    def test_conversational_rag_prompter_integration(self):
        """Test integration with ConversationalRAGPrompter."""
        prompter = ConversationalRAGPrompter(self.config["prompting"])
        
        # Mock the router's route_query method
        mock_decision = {
            "category": QueryCategory.RAG_FACTUAL.value,
            "strategy": ResponseStrategy.RAG_RETRIEVAL.value,
            "confidence": 0.9,
            "reasoning_chain": ["Need to retrieve billing information"],
            "response": "Mock response",
            "metadata": {"domain": "billing"}
        }
        
        with patch.object(prompter.router, 'route_query', return_value=mock_decision):
            query = "What is my billing cycle?"
            context = {"session_id": "test123"}
            
            # Test the prompter integration
            result = prompter.format(context={"query": query, "documents": []}, config={})
            
            # Should return a formatted prompt that incorporates routing decision
            assert isinstance(result, str)
            assert len(result) > 0

    def test_basic_routing_functionality(self):
        """Test basic routing with mocked LLM."""
        router = ConversationalRouter(self.config["prompting"]["routing_config"])
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            '{"topic": "billing", "is_broad": false, "is_ambiguous": false, "confidence": 0.9, "reasoning": "Specific billing question"}',
            '{"category": "rag_factual", "metadata": {}, "entities": ["billing"], "intent": "get_info", "confidence": 0.85, "reasoning": "Factual question requiring retrieval"}'
        ]
        
        with patch.object(router, '_load_prompt', side_effect=lambda x: self.mock_templates.get(x, f"Mock {x} template")):
            query = "What is my billing cycle?"
            result = router.route_query(query, "test_session")
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "strategy" in result
            assert "category" in result
            assert "confidence" in result
            assert result["category"] == "rag_factual"


class TestConversationalRoutingRealtimeSimulation:
    """Test realistic conversation scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "prompting": {
                "routing_config": {
                    "llm_config": {"type": "mock"},
                    "templates_dir": "templates/routing",
                    "max_conversation_history": 20
                }
            }
        }

    def test_simple_conversation_flow(self):
        """Test a simple conversation flow."""
        router = ConversationalRouter(self.config["prompting"]["routing_config"])
        mock_llm = Mock()
        router.set_llm(mock_llm)
        
        # Mock responses for greeting -> question -> farewell
        mock_llm.generate.side_effect = [
            # Greeting analysis
            '{"topic": "general", "is_broad": true, "is_ambiguous": false, "confidence": 0.8, "reasoning": "General greeting"}',
            '{"category": "greeting", "metadata": {}, "entities": [], "intent": "greet", "confidence": 0.9, "reasoning": "Simple greeting"}',
            # Question analysis  
            '{"topic": "product", "is_broad": false, "is_ambiguous": false, "confidence": 0.85, "reasoning": "Product inquiry"}',
            '{"category": "rag_factual", "metadata": {}, "entities": ["product"], "intent": "get_info", "confidence": 0.8, "reasoning": "Product question"}'
        ]
        
        with patch.object(router, '_load_prompt', return_value="Mock template"):
            # Test greeting
            result1 = router.route_query("Hi there!", "session1")
            assert result1["category"] == "greeting"
            
            # Test product question
            result2 = router.route_query("Tell me about your products", "session1")
            assert result2["category"] == "rag_factual"
            
            # Verify conversation context is maintained
            assert "session1" in router.conversation_contexts
            assert len(router.conversation_contexts["session1"].history) == 2
