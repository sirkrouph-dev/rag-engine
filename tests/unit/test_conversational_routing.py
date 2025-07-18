import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Union
import json

from rag_engine.core.conversational_routing import (
    ConversationalRouter,
    QueryCategory,
    ResponseStrategy,
    TopicAnalysis,
    QueryClassification,
    RoutingDecision,
    ConversationContext,
)

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.responses = {}

    def generate(self, prompt: str, **kwargs) -> str:
        """Mock the LLM's completion method."""
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return json.dumps({"error": "No mock response for prompt"})

    def set_next_response(self, key: str, response_data: Union[Dict, str]):
        """Set the next response for a given key."""
        if isinstance(response_data, dict):
            self.responses[key] = json.dumps(response_data)
        else:
            self.responses[key] = response_data

class TestConversationalRouter(unittest.TestCase):
    
    def setUp(self):
        """Set up a mock router for testing."""
        self.mock_llm = MockLLM()
        self.config = {
            "llm": {"provider": "mock", "config": {}},
            "routing_prompts_path": "templates/routing"
        }
        # We patch the _load_prompt method directly to avoid file system dependency
        with patch.object(ConversationalRouter, '_load_prompt', return_value="Mocked Prompt") as mock_load:
        self.router = ConversationalRouter(self.config)
        self.router.set_llm(self.mock_llm)
    
    def test_router_initialization(self):
        """Test router is initialized correctly."""
        self.assertIsNotNone(self.router)
        self.assertEqual(self.router.llm, self.mock_llm)
        self.assertIsInstance(self.router.conversation_contexts, dict)
    
    def test_topic_analysis_parsing(self):
        """Test that the router correctly parses the topic analysis from the LLM."""
        self.mock_llm.set_next_response("topic_analysis", {
            "topic": "machine_learning", "is_broad": True, "is_ambiguous": False, 
            "confidence": 0.9, "reasoning": "test"
        })
        context = self.router._get_context("test_session")
        # Manually set the prompt for the mock to catch it
        self.router.topic_analysis_prompt = "topic_analysis"
        analysis = self.router._analyze_topic("What is supervised learning?", context)
        self.assertIsInstance(analysis, TopicAnalysis)
        self.assertEqual(analysis.topic, "machine_learning")
    
    def test_query_classification_parsing(self):
        """Test that the router correctly parses the query classification from the LLM."""
        self.mock_llm.set_next_response("classification", {
            "category": "rag_factual", "metadata": {}, "entities": ["random_forests"], 
            "intent": "learn", "confidence": 0.9, "reasoning": "test"
        })
        topic_analysis = TopicAnalysis("test", False, False, 0.9, "test")
        context = self.router._get_context("test_session")
        self.router.classification_prompt = "classification"
        classification = self.router._classify_query("query", topic_analysis, context)
        self.assertEqual(classification.category, QueryCategory.RAG_FACTUAL)
    
    def test_strategy_selection(self):
        """Test the logic for selecting a response strategy."""
        classification = QueryClassification(QueryCategory.RAG_FACTUAL, {}, [], "test", 0.9, "test")
        topic_analysis = TopicAnalysis("test", False, False, 0.9, "test")
        strategy = self.router._determine_strategy(classification, topic_analysis)
        self.assertEqual(strategy, ResponseStrategy.RAG_RETRIEVAL)

    @patch('rag_engine.core.conversational_routing.ConversationalRouter._generate_response')
    def test_full_routing_pipeline(self, mock_generate_response):
        """Test the full, end-to-end routing pipeline."""
        mock_generate_response.return_value = {"content": "This is a RAG response."}
        self.mock_llm.set_next_response("topic_analysis", {"topic": "test"})
        self.mock_llm.set_next_response("classification", {"category": "rag_factual", "confidence": 0.9})
        self.router.topic_analysis_prompt = "topic_analysis"
        self.router.classification_prompt = "classification"
        
        decision = self.router.route_query("What is a transformer model?", "test_session")
        self.assertEqual(decision["strategy"], "rag_retrieval")
    
    def test_conversation_context_management(self):
        """Test that conversation context is created and updated correctly."""
        session_id = "context_session"
        with patch.object(self.router, '_generate_response', return_value={"content": "response"}):
             self.router.route_query("Query 1", session_id)
        self.assertIn(session_id, self.router.conversation_contexts)
        self.assertEqual(len(self.router.conversation_contexts[session_id].history), 1)
        with patch.object(self.router, '_generate_response', return_value={"content": "response"}):
            self.router.route_query("Query 2", session_id)
        self.assertEqual(len(self.router.conversation_contexts[session_id].history), 2)
    
    def test_error_handling_fallback(self):
        """Test the router's fallback mechanism."""
        self.mock_llm.generate = MagicMock(side_effect=Exception("LLM FAILED"))
        decision = self.router.route_query("This will fail", "fail_session")
        self.assertEqual(decision['strategy'], 'fallback')

if __name__ == '__main__':
    unittest.main() 