import unittest
from unittest.mock import Mock, patch, MagicMock
from rag_engine.core.conversational_routing import (
    ConversationalRouter, 
    QueryCategory, 
    ResponseStrategy,
    TopicAnalysis,
    QueryClassification,
    ConversationContext
)


class MockLLM:
    """Mock LLM for testing."""
    def __init__(self):
        self.call_count = 0
    
    def reset(self):
        """Reset the call count for new tests."""
        self.call_count = 0
    
    def generate(self, prompt, config=None, **kwargs):
        """Mock generate method that returns different responses based on prompt content."""
        self.call_count += 1
        
        # More robust prompt matching based on content
        prompt_lower = prompt.lower()
        
        # Classification - check this FIRST and be more specific
        if "classify the user's query" in prompt_lower and "json" in prompt_lower:
            return '{"category": "rag_factual", "entities": ["test"], "intent": "test_intent", "confidence": 0.9, "metadata": {}, "reasoning": "test classification"}'
        
        # Topic analysis - look for the actual prompt content
        elif "analyze the user's query" in prompt_lower and "json" in prompt_lower:
            return '{"topic": "test_topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "test reasoning"}'
        
        # Response generation
        elif any(keyword in prompt_lower for keyword in ["generate", "response", "answer"]):
            return "This is a test response from the mock LLM."
        
        # Default fallback - return topic analysis format if it looks like a JSON request
        elif "json" in prompt_lower:
            return '{"topic": "test_topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "fallback topic analysis"}'
        
        # Final fallback
        else:
            return "Mock LLM response"


class TestConversationalRouter(unittest.TestCase):
    """Test cases for ConversationalRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLM()
        self.config = {
            "llm": {"provider": "mock", "config": {}},
            "routing_prompts_path": "templates/routing"
        }
        
        # Create proper mock prompts that match what the implementation expects
        def mock_load_prompt(prompt_type):
            if prompt_type == "topic_analysis":
                return """Analyze the user's query to understand the topic.
                
Query: {query}
Context: {context}

Please respond with JSON in this format:
{{"topic": "main topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "explanation"}}"""
            elif prompt_type == "query_classification":
                return """Classify the user's query and extract metadata.
                
Query: {query}
Topic: {topic}

Please respond with JSON in this format:
{{"category": "rag_factual", "entities": [], "intent": "intent", "confidence": 0.9, "metadata": {{}}, "reasoning": "explanation"}}"""
            else:
                return "Generate response for {query}"
        
        # Patch the _load_prompt method with our proper mock prompts
        with patch.object(ConversationalRouter, '_load_prompt', side_effect=mock_load_prompt) as mock_load:
            self.router = ConversationalRouter(self.config)
            self.router.set_llm(self.mock_llm)
            
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'mock_llm'):
            self.mock_llm.reset()
    
    def test_router_initialization(self):
        """Test router is initialized correctly."""
        self.assertIsNotNone(self.router)
        self.assertEqual(self.router.config, self.config)
        self.assertEqual(self.router.llm, self.mock_llm)
    
    @patch.object(ConversationalRouter, '_load_prompt')
    def test_topic_analysis(self, mock_load):
        """Test topic analysis functionality."""
        # Set up proper mock prompts
        def mock_load_prompt_side_effect(prompt_type):
            if prompt_type == "topic_analysis":
                return """Analyze the user's query to understand the topic.
                
Query: {query}
Context: {context}

Please respond with JSON in this format:
{{"topic": "main topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "explanation"}}"""
            elif prompt_type == "query_classification":
                return """Classify the user's query and extract metadata.
                
Query: {query}
Topic: {topic}

Please respond with JSON in this format:
{{"category": "rag_factual", "entities": [], "intent": "intent", "confidence": 0.9, "metadata": {{}}, "reasoning": "explanation"}}"""
            else:
                return "Generate response for {query}"
        
        mock_load.side_effect = mock_load_prompt_side_effect
        
        # Create a fresh router for this test
        fresh_router = ConversationalRouter(self.config)
        fresh_router.set_llm(self.mock_llm)
        
        query = "What is the pricing for your service?"
        context = ConversationContext(
            history=[],
            accumulated_metadata={},
            chain_of_thought=[],
            user_profile={},
            session_id="test_session"
        )
        
        analysis = fresh_router._analyze_topic(query, context)
        
        self.assertIsInstance(analysis, TopicAnalysis)
        self.assertEqual(analysis.topic, "test_topic")
        self.assertEqual(analysis.confidence, 0.8)
    
    @patch.object(ConversationalRouter, '_load_prompt')
    def test_query_classification(self, mock_load):
        """Test query classification functionality."""
        # Set up proper mock prompts
        def mock_load_prompt_side_effect(prompt_type):
            if prompt_type == "topic_analysis":
                return """Analyze the user's query to understand the topic.
                
Query: {query}
Context: {context}

Please respond with JSON in this format:
{{"topic": "main topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "explanation"}}"""
            elif prompt_type == "query_classification":
                return """Classify the user's query and extract metadata.
                
Query: {query}
Topic: {topic}

Please respond with JSON in this format:
{{"category": "rag_factual", "entities": [], "intent": "intent", "confidence": 0.9, "metadata": {{}}, "reasoning": "explanation"}}"""
            else:
                return "Generate response for {query}"
        
        mock_load.side_effect = mock_load_prompt_side_effect
        
        # Create a fresh router for this test
        fresh_router = ConversationalRouter(self.config)
        fresh_router.set_llm(self.mock_llm)
        
        query = "What is the pricing?"
        topic_analysis = TopicAnalysis(
            topic="pricing",
            is_broad=False,
            is_ambiguous=False,
            confidence=0.8,
            reasoning="Clear pricing question"
        )
        context = ConversationContext(
            history=[],
            accumulated_metadata={},
            chain_of_thought=[],
            user_profile={},
            session_id="test_session"
        )
        
        classification = fresh_router._classify_query(query, topic_analysis, context)
        
        self.assertIsInstance(classification, QueryClassification)
        self.assertEqual(classification.category, QueryCategory.RAG_FACTUAL)
        self.assertEqual(classification.confidence, 0.9)
    
    @patch.object(ConversationalRouter, '_load_prompt')
    def test_full_routing_pipeline(self, mock_load):
        """Test the complete routing pipeline."""
        # Set up proper mock prompts
        def mock_load_prompt_side_effect(prompt_type):
            if prompt_type == "topic_analysis":
                return """Analyze the user's query to understand the topic.
                
Query: {query}
Context: {context}

Please respond with JSON in this format:
{{"topic": "main topic", "is_broad": false, "is_ambiguous": false, "confidence": 0.8, "reasoning": "explanation"}}"""
            elif prompt_type == "query_classification":
                return """Classify the user's query and extract metadata.
                
Query: {query}
Topic: {topic}

Please respond with JSON in this format:
{{"category": "rag_factual", "entities": [], "intent": "intent", "confidence": 0.9, "metadata": {{}}, "reasoning": "explanation"}}"""
            else:
                return "Generate response for {query}"
        
        mock_load.side_effect = mock_load_prompt_side_effect
        
        # Create a fresh router for this test
        fresh_router = ConversationalRouter(self.config)
        fresh_router.set_llm(self.mock_llm)
        
        query = "What are your pricing plans?"
        session_id = "test_session"
        
        result = fresh_router.route_query(query, session_id)
        
        self.assertIn("response", result)
        self.assertIn("strategy", result)
        self.assertIn("category", result)
        self.assertIn("confidence", result)


class TestConversationalRAGPrompter(unittest.TestCase):
    """Test cases for ConversationalRAGPrompter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "enable_routing": True,
            "fallback_to_simple": True
        }
        
    @patch('rag_engine.core.conversational_integration.ConversationalRouter')
    def test_prompter_initialization(self, mock_router_class):
        """Test prompter initialization."""
        from rag_engine.core.conversational_integration import ConversationalRAGPrompter
        
        prompter = ConversationalRAGPrompter(self.config)
        self.assertEqual(prompter.config, self.config)
        self.assertTrue(prompter.enable_routing)


if __name__ == '__main__':
    unittest.main() 