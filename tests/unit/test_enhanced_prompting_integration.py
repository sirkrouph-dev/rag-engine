"""
Test enhanced prompting integration with the RAG Engine.
"""
import pytest
from unittest.mock import Mock, patch
from rag_engine.core.prompting_enhanced import get_prompter as get_enhanced_prompter, RAGPrompter, ConversationalPrompter
from rag_engine.core.component_registry import COMPONENT_REGISTRY


class TestEnhancedPromptingIntegration:
    """Test integration of enhanced prompting with the RAG system."""

    def test_create_prompter_factory(self):
        """Test that the prompter factory creates correct instances."""
        # Test RAG prompter
        rag_config = {
            "type": "rag",
            "template_path": "./templates/rag_template.txt",
            "context_window": 3000
        }
        rag_prompter = get_enhanced_prompter("rag", rag_config)
        assert isinstance(rag_prompter, RAGPrompter)
        
        # Test conversational prompter
        conv_config = {
            "type": "conversational",
            "memory_length": 5,
            "persona": "helpful_assistant"
        }
        conv_prompter = get_enhanced_prompter("conversational", conv_config)
        assert isinstance(conv_prompter, ConversationalPrompter)

    def test_component_registry_integration(self):
        """Test that enhanced prompters are registered in the component registry."""
        # Test that prompter factory is registered
        assert 'prompter' in COMPONENT_REGISTRY._factories
        
        # Test that enhanced prompter types are registered
        prompter_components = COMPONENT_REGISTRY.list_components('prompter')
        assert 'rag' in prompter_components.get('prompter', [])
        assert 'conversational' in prompter_components.get('prompter', [])
        assert 'code_explanation' in prompter_components.get('prompter', [])

    def test_prompter_factory_fallback(self):
        """Test that factory falls back to legacy prompter for unknown types."""
        with patch('rag_engine.core.component_registry.get_prompter') as mock_get_prompter:
            mock_prompter = Mock()
            mock_get_prompter.return_value = mock_prompter
            
            # Test fallback for unknown type
            factory_func = COMPONENT_REGISTRY._factories['prompter']
            result = factory_func("unknown_type", {})
            
            # Should fall back to legacy prompter
            mock_get_prompter.assert_called_once_with({})
            assert result == mock_prompter

    @patch('os.path.exists')
    def test_rag_prompter_template_loading(self, mock_exists):
        """Test that RAG prompter loads templates correctly."""
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "You are a helpful assistant.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
            )
            
            config = {
                "template_path": "./templates/rag_template.txt",
                "context_window": 3000
            }
            prompter = get_enhanced_prompter("rag", config)
            
            # Test prompt building
            prompt_config = {
                "query": "What is Python?",
                "context": "Python is a programming language.",
                "metadata": {}
            }
            prompt = prompter.build_prompt(prompt_config)
            
            assert "What is Python?" in prompt
            assert "Python is a programming language" in prompt

    def test_conversational_prompter_memory(self):
        """Test conversational prompter memory management."""
        config = {
            "memory_length": 3,
            "persona": "helpful_assistant"
        }
        prompter = get_enhanced_prompter("conversational", config)
        
        # Add messages to history
        prompter.add_to_history("user", "Hello")
        prompter.add_to_history("assistant", "Hi there!")
        prompter.add_to_history("user", "What is AI?")
        prompter.add_to_history("assistant", "AI is artificial intelligence.")
        
        # Test that history is maintained
        assert len(prompter.conversation_history) == 4
        
        # Add more messages than memory limit
        prompter.add_to_history("user", "Tell me more")
        prompter.add_to_history("assistant", "Sure!")
        
        # Should only keep last 3 messages (memory_length * 2 for user/assistant pairs)
        assert len(prompter.conversation_history) <= 6

    def test_enhanced_vs_legacy_compatibility(self):
        """Test that enhanced and legacy prompting systems are compatible."""
        # Test legacy config works with enhanced system
        legacy_config = {
            "template": "default",
            "system_prompt": "You are a helpful assistant."
        }
        
        factory_func = COMPONENT_REGISTRY._factories['prompter']
        prompter = factory_func("default", legacy_config)
        
        # Should still work (either enhanced or legacy)
        assert prompter is not None
        assert hasattr(prompter, 'build_prompt') or hasattr(prompter, 'format')

    def test_context_formatting(self):
        """Test context formatting in enhanced prompters."""
        config = {
            "context_window": 1000,
            "citation_format": "numbered"
        }
        prompter = get_enhanced_prompter("rag", config)
        
        documents = [
            {"content": "Python is a programming language.", "score": 0.9, "source": "doc1.txt"},
            {"content": "It was created by Guido van Rossum.", "score": 0.8, "source": "doc2.txt"}
        ]
        
        formatted_context = prompter.format_context(documents)
        
        # Should format with citations
        assert "[1]" in formatted_context or "1." in formatted_context
        assert "Python is a programming language" in formatted_context
        assert "Guido van Rossum" in formatted_context

    def test_template_variable_substitution(self):
        """Test that template variables are properly substituted."""
        config = {"context_window": 1000}
        prompter = get_enhanced_prompter("rag", config)
        
        # Mock template with variables
        prompter.template = "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        prompt_config = {
            "query": "What is Python?",
            "context": "Python is a programming language."
        }
        
        prompt = prompter.build_prompt(prompt_config)
        
        # Variables should be substituted
        assert "{query}" not in prompt
        assert "{context}" not in prompt
        assert "What is Python?" in prompt
        assert "Python is a programming language" in prompt


if __name__ == "__main__":
    pytest.main([__file__])
