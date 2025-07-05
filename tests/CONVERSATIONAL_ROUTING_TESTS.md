# Conversational Routing System Test Documentation

This document describes the comprehensive test suite for the conversational routing system in the RAG Engine.

## Test Structure

### Unit Tests

#### `test_conversational_routing.py`
**Location**: `tests/unit/test_conversational_routing.py`

Core unit tests for the conversational routing system components:

- **TestConversationalRouter**: Tests the core routing logic
  - Router initialization and configuration
  - Topic analysis functionality
  - Query classification 
  - Full routing pipeline
  - Error handling and edge cases
  - Template loading and management

- **TestConversationalRAGPrompter**: Tests the prompter integration
  - Prompter initialization with routing
  - Prompt generation with routing decisions
  - Fallback mechanisms
  - Template usage and customization

#### `test_conversational_routing_advanced.py`
**Location**: `tests/unit/test_conversational_routing_advanced.py`

Advanced unit tests for edge cases, performance, and stress scenarios:

- **TestConversationalRoutingEdgeCases**: Edge case handling
  - Extremely long queries
  - Unicode and special characters
  - Malformed JSON responses from LLM
  - Concurrent routing requests
  - Memory limit scenarios
  - Invalid confidence scores
  - Missing required fields

- **TestConversationalRoutingPerformance**: Performance testing
  - Routing latency measurements
  - Throughput under load
  - Memory usage stability
  - Large context performance

- **TestConversationalRoutingStressTest**: Stress testing
  - Rapid-fire requests
  - Memory pressure scenarios
  - Long-running stability

### Integration Tests

#### `test_conversational_routing_api.py`
**Location**: `tests/integration/test_conversational_routing_api.py`

API endpoint integration tests:

- **TestConversationalRoutingAPI**: API endpoint testing
  - Template management endpoints (`GET/PUT /routing/templates/*`)
  - Configuration endpoints (`GET/PUT /routing/config`)
  - Testing endpoints (`POST /routing/test`)
  - Analytics endpoints (`GET /routing/analytics`)
  - Error handling and validation

- **TestConversationalRoutingIntegration**: System integration
  - Router integration with RAG pipeline
  - Template validation
  - Error handling in routing pipeline
  - Context management across queries
  - Routing decision serialization

#### `test_conversational_routing_e2e.py`
**Location**: `tests/integration/test_conversational_routing_e2e.py`

End-to-end workflow testing:

- **TestConversationalRoutingE2E**: Complete workflow tests
  - RAG-suitable query workflow
  - Greeting workflow
  - Out-of-context query workflow
  - Conversation context evolution
  - Error recovery mechanisms
  - Low confidence handling
  - Template customization

- **TestConversationalRoutingRealtimeSimulation**: Real-time scenarios
  - Customer support conversation simulation
  - Context switching conversations
  - Ambiguous query handling

## Test Configuration

### Test Fixtures

#### `conftest_routing.py`
**Location**: `tests/conftest_routing.py`

Comprehensive pytest fixtures for routing tests:

- **Configuration Fixtures**:
  - `basic_routing_config`: Basic routing configuration
  - `customer_support_config`: Customer support domain config
  - `performance_config`: Performance-optimized config

- **Mock Fixtures**:
  - `mock_llm`: Mock LLM with configurable responses
  - `mock_router`: Pre-configured mock router
  - `mock_templates`: Standard routing templates
  - `mock_prompter`: Mock conversational prompter

- **Data Fixtures**:
  - `sample_routing_decision`: Example routing decision
  - `conversation_history`: Sample conversation data
  - `test_queries`: Categorized test queries
  - `temp_templates_dir`: Temporary template directory

#### Test Configuration Files

**Location**: `tests/configs/conversational_routing_test_configs.json`

JSON configuration files for different test scenarios:
- `test_conversational_routing`: General testing configuration
- `test_customer_support`: Customer support domain testing
- `test_performance`: Performance testing configuration
- `test_minimal`: Minimal configuration for basic tests

## Running Tests

### Test Runner Script

**Location**: `tests/run_routing_tests.py`

Comprehensive test runner with multiple options:

```bash
# Run all tests
python tests/run_routing_tests.py all

# Run specific test types
python tests/run_routing_tests.py unit
python tests/run_routing_tests.py integration
python tests/run_routing_tests.py api
python tests/run_routing_tests.py e2e
python tests/run_routing_tests.py performance

# Run with options
python tests/run_routing_tests.py all --verbose --coverage
python tests/run_routing_tests.py unit --quick  # Excludes slow tests

# Run specific test class
python tests/run_routing_tests.py --class TestConversationalRouter

# Generate comprehensive report
python tests/run_routing_tests.py report
```

### Direct pytest Commands

```bash
# Run all routing tests
pytest tests/unit/test_conversational_routing*.py tests/integration/test_conversational_routing*.py -v

# Run with coverage
pytest tests/unit/test_conversational_routing*.py --cov=rag_engine.core.conversational_routing --cov-report=html

# Run specific test categories
pytest -m "unit" tests/
pytest -m "integration" tests/
pytest -m "performance" tests/
pytest -m "not slow" tests/  # Exclude slow tests

# Run specific test files
pytest tests/unit/test_conversational_routing.py::TestConversationalRouter::test_router_initialization -v
```

## Test Categories and Markers

### Pytest Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.stress`: Stress tests
- `@pytest.mark.slow`: Slow-running tests

### Test Categories

1. **Core Functionality Tests**
   - Basic routing logic
   - Template management
   - Configuration handling
   - LLM integration

2. **Integration Tests**
   - API endpoints
   - Pipeline integration
   - Database interactions
   - Component interactions

3. **Workflow Tests**
   - Complete user scenarios
   - Multi-turn conversations
   - Error recovery flows
   - Real-time simulations

4. **Performance Tests**
   - Latency measurements
   - Throughput testing
   - Memory usage
   - Concurrent load

5. **Edge Case Tests**
   - Input validation
   - Error handling
   - Boundary conditions
   - Malformed data

## Test Data and Scenarios

### Query Categories

The tests cover various query types:

- **Greetings**: "Hello!", "Hi there!", "Good morning!"
- **RAG Factual**: "What is the pricing?", "How do I reset my password?"
- **Out of Context**: "What's the weather?", "Tell me a joke"
- **Follow-up**: "Can you explain more?", "What about the alternative?"
- **Ambiguous**: "It doesn't work", "Help", "I'm confused"

### Conversation Scenarios

- **Customer Support**: Billing issues, technical problems, account management
- **Topic Switching**: Conversations that change topics mid-stream
- **Long Conversations**: Extended multi-turn conversations
- **Context Building**: Conversations that build context over time

### Error Scenarios

- **LLM Failures**: Service unavailable, timeout, malformed responses
- **Low Confidence**: Uncertain classifications, ambiguous queries
- **Invalid Input**: Empty queries, extreme lengths, invalid characters
- **System Errors**: Memory pressure, concurrent access, resource limits

## Coverage Goals

### Code Coverage Targets

- **Core Routing Logic**: 95%+ coverage
- **Template Management**: 90%+ coverage
- **API Endpoints**: 85%+ coverage
- **Error Handling**: 80%+ coverage

### Scenario Coverage

- **Query Types**: All major query categories
- **Conversation Flows**: Common user workflows
- **Error Conditions**: Major failure modes
- **Performance Scenarios**: Typical load patterns

## Continuous Integration

### CI/CD Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Conversational Routing Tests
  run: |
    python tests/run_routing_tests.py all --coverage
    python tests/run_routing_tests.py performance --quick
```

### Test Environment Requirements

- **Python 3.8+**
- **pytest 6.0+**
- **Mock LLM**: No external LLM service required for basic tests
- **Memory**: Minimum 2GB for stress tests
- **Disk**: 100MB for test artifacts

## Debugging and Troubleshooting

### Common Issues

1. **Template Loading Failures**
   - Check template directory paths
   - Verify template file permissions
   - Validate template syntax

2. **Mock LLM Issues**
   - Ensure mock responses are valid JSON
   - Check response format matches expected schema
   - Verify mock configuration

3. **Performance Test Failures**
   - Check system resources
   - Adjust performance thresholds
   - Consider test environment differences

### Debug Commands

```bash
# Run tests with detailed output
pytest tests/unit/test_conversational_routing.py -vvv --tb=long

# Run single test with debugging
pytest tests/unit/test_conversational_routing.py::TestConversationalRouter::test_router_initialization -vvv --pdb

# Check test discovery
pytest --collect-only tests/
```

## Extending Tests

### Adding New Test Cases

1. **Identify Test Category**: Unit, integration, or e2e
2. **Create Test Method**: Follow naming convention `test_*`
3. **Use Appropriate Fixtures**: Leverage existing fixtures
4. **Add Markers**: Use `@pytest.mark.*` for categorization
5. **Document Test**: Add docstring explaining test purpose

### Adding New Test Files

1. **Follow Naming Convention**: `test_*.py`
2. **Import Required Fixtures**: From `conftest_routing.py`
3. **Organize by Functionality**: Group related tests in classes
4. **Update Test Runner**: Add to `run_routing_tests.py` if needed

### Custom Fixtures

Create domain-specific fixtures for specialized testing:

```python
@pytest.fixture
def healthcare_config():
    """Configuration for healthcare domain testing."""
    return {
        "domain_config": {
            "name": "healthcare",
            "topics": ["symptoms", "medications", "appointments"]
        }
    }
```

## Performance Benchmarks

### Expected Performance

- **Routing Latency**: < 100ms per query (mock LLM)
- **Throughput**: > 50 queries/second (concurrent)
- **Memory Growth**: < 10% over 1000 queries
- **Template Loading**: < 1 second for 100 templates

### Performance Test Results

Run performance tests to establish baselines:

```bash
python tests/run_routing_tests.py performance --verbose
```

Results are output to console and can be used to track performance regression.

---

This comprehensive test suite ensures the conversational routing system is robust, performant, and reliable across various scenarios and use cases.
