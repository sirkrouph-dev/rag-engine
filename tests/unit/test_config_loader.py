"""
Unit tests for configuration loading and validation.
"""
import pytest
import json
import os
from unittest.mock import patch
from rag_engine.config.loader import load_config, substitute_env_vars
from rag_engine.config.schema import RAGConfig


class TestConfigLoader:
    """Test configuration loading functionality."""

    def test_load_json_config(self, temp_dir, sample_config):
        """Test loading a valid JSON configuration file."""
        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(sample_config, f)
        
        config = load_config(config_path)
        
        assert isinstance(config, RAGConfig)
        assert len(config.documents) == 1
        assert config.chunking.method == "fixed"
        assert config.embedding.provider == "huggingface"

    def test_load_yaml_config(self, temp_dir, sample_config):
        """Test loading a valid YAML configuration file."""
        import yaml
        
        config_path = os.path.join(temp_dir, "test_config.yml")
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        config = load_config(config_path)
        
        assert isinstance(config, RAGConfig)
        assert config.chunking.method == "fixed"

    def test_environment_variable_substitution_function(self):
        """Test the environment variable substitution function directly."""
        config_with_env = {
            "embedding": {
                "provider": "openai",
                "api_key": "${TEST_API_KEY}"
            },
            "nested": {
                "value": "${TEST_NESTED_VALUE}"
            }
        }
        
        with patch.dict(os.environ, {
            "TEST_API_KEY": "test_key_123",
            "TEST_NESTED_VALUE": "nested_value_456"
        }):
            result = substitute_env_vars(config_with_env)
        
        assert result["embedding"]["api_key"] == "test_key_123"
        assert result["nested"]["value"] == "nested_value_456"

    def test_missing_environment_variable(self):
        """Test behavior when environment variable is missing."""
        config_with_missing_env = {
            "embedding": {
                "api_key": "${MISSING_API_KEY}"
            }
        }
        
        result = substitute_env_vars(config_with_missing_env)
        
        # Should leave the placeholder as-is when env var is missing
        assert result["embedding"]["api_key"] == "${MISSING_API_KEY}"

    def test_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_invalid_json_config(self, temp_dir):
        """Test error handling for invalid JSON."""
        config_path = os.path.join(temp_dir, "invalid_config.json")
        with open(config_path, "w") as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_config(config_path)

    def test_config_validation_success(self, sample_config_file):
        """Test that valid config passes Pydantic validation."""
        config = load_config(sample_config_file)
        
        # Should return a valid RAGConfig object
        assert isinstance(config, RAGConfig)
        assert config.chunking.method == "fixed"
        assert config.embedding.provider == "huggingface"

    def test_partial_config_validation_error(self, temp_dir):
        """Test validation error for incomplete config."""
        minimal_config = {
            "documents": [{"type": "txt", "path": "./test.txt"}],
            "chunking": {"method": "fixed", "max_tokens": 200},
            "embedding": {"provider": "huggingface", "model": "test-model"},
            # Missing required fields: vectorstore, retrieval, prompting, llm, output
        }
        
        config_path = os.path.join(temp_dir, "incomplete_config.json")
        with open(config_path, "w") as f:
            json.dump(minimal_config, f)
        
        # Should raise validation error for missing required fields
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_config(config_path)
