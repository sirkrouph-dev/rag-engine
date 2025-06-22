# Tests

This directory contains all test files for the RAG Engine.

## 📁 Directory Structure

```
tests/
├── configs/          # Test configuration files
├── fixtures/         # Test data and fixtures
├── integration/      # Integration tests
├── unit/            # Unit tests
├── conftest.py      # Pytest configuration
├── __init__.py      # Test package init
└── *.py             # Individual test modules
```

## 🧪 Running Tests

### All Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest --cov=rag_engine tests/
```

### Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Specific test file
python -m pytest tests/test_embedder.py -v
```

### Test Types

#### Unit Tests (`unit/`)
- Test individual components in isolation
- Fast execution
- No external dependencies

#### Integration Tests (`integration/`)
- Test component interactions
- End-to-end functionality
- May require API keys for external services

#### Configuration Tests (`configs/`)
- Various test configurations
- Different embedding providers
- Local and cloud setups

## 🔧 Test Configuration

Test configurations are provided for different scenarios:
- `test_config_simple.json` - Basic test setup
- `test_config_local.json` - Local-only dependencies
- `test_config_local_llm.json` - Local LLM testing

## 📊 Current Status

- ✅ **59/59 Tests Passing**
- ✅ **Zero Warnings**
- ✅ **100% Success Rate**
- ✅ **Comprehensive Coverage**

## 🔍 Test Guidelines

When adding new tests:

1. **Unit tests** for new components
2. **Integration tests** for workflows
3. **Mock external services** when possible
4. **Use test configurations** from `configs/`
5. **Follow pytest conventions**

## ⚠️ Notes

- Some tests require API keys (set in environment)
- Tests marked with `@pytest.mark.integration` may be slower
- Use `pytest -m "not integration"` to skip integration tests
