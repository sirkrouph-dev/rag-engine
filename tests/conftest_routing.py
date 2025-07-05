"""
Pytest configuration and fixtures for conversational routing tests.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import routing components
from rag_engine.core.conversational_routing import (
    ConversationalRouter,
    QueryCategory,
    ResponseStrategy,
    TopicAnalysis,
    QueryClassification,
    RoutingDecision
)
from rag_engine.core.conversational_integration import ConversationalRAGPrompter


@pytest.fixture
def basic_routing_config():
    """Basic routing configuration for tests."""
    return {
        "topic_analysis_temperature": 0.1,
        "classification_temperature": 0.1,
        "response_temperature": 0.7,
        "max_conversation_history": 10,
        "confidence_threshold": 0.8,
        "enable_reasoning_chain": True
    }


@pytest.fixture
def customer_support_config():
    """Customer support specific configuration."""
    return {
        "prompting": {
            "type": "conversational_rag",
            "enable_routing": True,
            "fallback_to_simple": True,
            "routing_config": {
                "topic_analysis_temperature": 0.1,
                "classification_temperature": 0.1,
                "response_temperature": 0.7,
                "max_conversation_history": 20,
                "confidence_threshold": 0.75,
                "enable_reasoning_chain": True
            },
            "domain_config": {
                "name": "customer_support",
                "description": "Customer support for a SaaS platform",
                "topics": ["billing", "technical_issues", "account_management"],
                "specialization": "B2B SaaS customer support"
            },
            "system_prompt": "You are a professional customer support assistant."
        }
    }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    
    # Default response for topic analysis
    llm.invoke.return_value = json.dumps({
        "topic": "general",
        "confidence": 0.8,
        "reasoning": "Mock topic analysis"
    })
    
    return llm


@pytest.fixture
def mock_router(basic_routing_config, mock_llm):
    """Mock router with basic configuration."""
    router = ConversationalRouter(basic_routing_config)
    router.set_llm(mock_llm)
    return router


@pytest.fixture
def mock_templates():
    """Mock routing templates for testing."""
    return {
        "topic_analysis_template": """Analyze the topic of this query.

Query: {query}
Context: {context}

Respond with JSON containing topic, confidence, and reasoning.""",

        "query_classification_template": """Classify this query.

Query: {query}
Topic Analysis: {topic_analysis}
Context: {context}

Respond with JSON containing category, confidence, and reasoning.""",

        "rag_response_template": """Answer using retrieved information.

Retrieved Information: {retrieved_context}
User Question: {query}
Context: {conversation_context}

Provide a helpful response.""",

        "contextual_chat_template": """Continue the conversation naturally.

Context: {conversation_context}
Query: {query}

Respond conversationally.""",

        "polite_rejection_template": """Politely indicate the query is out of scope.

Query: {query}

Explain what you can help with instead.""",

        "clarification_request_template": """Request clarification for ambiguous query.

Query: {query}
Context: {conversation_context}

Ask for clarification."""
    }


@pytest.fixture
def sample_routing_decision():
    """Sample routing decision for testing."""
    return RoutingDecision(
        category=QueryCategory.RAG_FACTUAL,
        strategy=ResponseStrategy.RAG_RETRIEVAL,
        confidence=0.85,
        reasoning="Factual question requiring document retrieval",
        template_name="rag_response_template",
        context_updates={"domain": "test"}
    )


@pytest.fixture
def conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "I have a question about billing."},
        {"role": "assistant", "content": "I'd be happy to help with billing questions. What specifically would you like to know?"}
    ]


@pytest.fixture
def test_queries():
    """Sample test queries for different scenarios."""
    return {
        "greeting": [
            "Hello!",
            "Hi there!",
            "Good morning!",
            "How are you?"
        ],
        "rag_factual": [
            "What is the pricing for the premium plan?",
            "How do I reset my password?",
            "What are the system requirements?",
            "How does the backup feature work?"
        ],
        "out_of_context": [
            "What's the weather like today?",
            "How do I cook pasta?",
            "What's the capital of France?",
            "Tell me a joke."
        ],
        "follow_up": [
            "Can you explain that in more detail?",
            "What about the other option?",
            "How much does that cost?",
            "Is there an alternative?"
        ],
        "ambiguous": [
            "It doesn't work",
            "Help",
            "I'm confused",
            "This is broken",
            "What do I do?"
        ]
    }


@pytest.fixture
def temp_templates_dir():
    """Temporary directory with test templates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        templates_dir = Path(temp_dir) / "templates" / "routing"
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test templates
        test_templates = {
            "topic_analysis_template.txt": "Topic: {query}\nContext: {context}",
            "query_classification_template.txt": "Classify: {query}",
            "rag_response_template.txt": "Answer: {retrieved_context}\nQuery: {query}",
            "contextual_chat_template.txt": "Chat: {query}",
            "polite_rejection_template.txt": "Sorry: {query}",
            "clarification_request_template.txt": "Clarify: {query}"
        }
        
        for filename, content in test_templates.items():
            template_file = templates_dir / filename
            template_file.write_text(content)
        
        yield templates_dir


@pytest.fixture
def mock_prompter(customer_support_config):
    """Mock conversational RAG prompter."""
    return ConversationalRAGPrompter(customer_support_config["prompting"])


@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing."""
    return {
        "topic_analysis_temperature": 0.0,
        "classification_temperature": 0.0,
        "response_temperature": 0.3,
        "max_conversation_history": 3,
        "confidence_threshold": 0.9,
        "enable_reasoning_chain": False
    }


# Test markers for organizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add markers based on test name
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        if "stress" in item.name.lower():
            item.add_marker(pytest.mark.stress)
        if "slow" in item.name.lower() or "long_running" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Utility functions for tests
def create_mock_decision(category=QueryCategory.RAG_FACTUAL, 
                        strategy=ResponseStrategy.RAG_RETRIEVAL,
                        confidence=0.8):
    """Create a mock routing decision."""
    return RoutingDecision(
        category=category,
        strategy=strategy,
        confidence=confidence,
        reasoning=f"Mock decision for {category.value}",
        template_name=f"{strategy.value}_template",
        context_updates={"mock": True}
    )


def create_test_conversation(length=5):
    """Create a test conversation history."""
    conversation = []
    for i in range(length):
        conversation.extend([
            {"role": "user", "content": f"User message {i}"},
            {"role": "assistant", "content": f"Assistant response {i}"}
        ])
    return conversation


def assert_valid_routing_decision(decision):
    """Assert that a routing decision is valid."""
    assert isinstance(decision, RoutingDecision)
    assert isinstance(decision.category, QueryCategory)
    assert isinstance(decision.strategy, ResponseStrategy)
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.reasoning, str)
    assert len(decision.reasoning) > 0
    assert isinstance(decision.template_name, str)
    assert len(decision.template_name) > 0
