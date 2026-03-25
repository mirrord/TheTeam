# Test Fixtures and Configurations

This directory contains unit tests for the pithos module.

## Structure

- `test_agent.py` - Tests for agent and context management
- `test_flownode.py` - Tests for flow nodes (Prompt, Custom)
- `test_conditions.py` - Tests for conditions (Always, Count, Regex)
- `test_flowchart.py` - Tests for flowchart execution
- `test_config_manager.py` - Tests for configuration management

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_agent.py
```

Run with coverage:
```bash
pytest --cov=src/pithos --cov-report=html
```

Run specific test class or function:
```bash
pytest tests/test_agent.py::TestAgentContext::test_copy_context
```

## Test Categories

Tests are organized by module and functionality:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test interactions between components
- **Mock tests**: Tests that mock external dependencies (like LLM calls)

## Conventions

- All test files start with `test_`
- Test classes start with `Test`
- Test methods start with `test_`
- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Use fixtures for common setup
