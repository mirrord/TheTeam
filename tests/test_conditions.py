"""Unit tests for conditions module."""

import pytest
from pithos.conditions import (
    Edge,
    Condition,
    AlwaysCondition,
    CountCondition,
    RegexCondition,
    ConditionManager,
)
from unittest.mock import patch


class TestEdge:
    """Test Edge base class."""

    def test_edge_creation_default(self):
        edge = Edge()
        assert edge.enabled is False

    def test_edge_creation_enabled(self):
        edge = Edge(enabled=True)
        assert edge.enabled is True

    def test_is_open_false(self):
        edge = Edge(enabled=False)
        assert edge.is_open({}) is False

    def test_is_open_true(self):
        edge = Edge(enabled=True)
        assert edge.is_open({}) is True

    def test_traverse_returns_state(self):
        edge = Edge()
        state = {"key": "value"}
        result = edge.traverse(state)
        assert result is state


class TestCondition:
    """Test Condition class."""

    def test_condition_creation(self):
        cond = Condition(condition=lambda x: True)
        assert cond.is_open({}) is True

    def test_condition_with_lambda(self):
        cond = Condition(condition=lambda state: state.get("ready", False))
        assert cond.is_open({"ready": True}) is True
        assert cond.is_open({"ready": False}) is False

    def test_condition_with_function(self):
        def check_value(state):
            return state.get("value", 0) > 10

        cond = Condition(condition=check_value)
        assert cond.is_open({"value": 15}) is True
        assert cond.is_open({"value": 5}) is False

    def test_condition_registered_name(self):
        cond = Condition(condition=lambda x: True, registered_name="TestCondition")
        assert cond.registered_name == "TestCondition"

    def test_condition_str(self):
        cond = Condition(condition=lambda x: True, registered_name="MyCondition")
        str_repr = str(cond)
        assert "MyCondition" in str_repr

    def test_condition_to_dict(self):
        cond = Condition(condition=lambda x: True, registered_name="TestCond")
        d = cond.to_dict()
        assert d["type"] == "TestCond"

    def test_condition_from_dict_simple(self):
        data = {"condition": "lambda state: state.get('ok', False)"}
        cond = Condition.from_dict(data)
        assert cond.is_open({"ok": True}) is True
        assert cond.is_open({"ok": False}) is False


class TestAlwaysCondition:
    """Test AlwaysCondition."""

    def test_always_condition_always_true(self):
        assert AlwaysCondition.is_open({}) is True
        assert AlwaysCondition.is_open({"any": "state"}) is True

    def test_always_condition_registered_name(self):
        assert AlwaysCondition.registered_name == "AlwaysCondition"


class TestCountCondition:
    """Test CountCondition for counting traversals."""

    def test_count_condition_creation(self):
        cond = CountCondition(limit=3)
        assert cond.limit == 3
        assert cond._counter == 0

    def test_count_condition_default_limit(self):
        cond = CountCondition()
        assert cond.limit == 1

    def test_count_condition_is_open_under_limit(self):
        cond = CountCondition(limit=3)
        assert cond.is_open({}) is True

    def test_count_condition_is_open_at_limit(self):
        cond = CountCondition(limit=1)
        assert cond.is_open({}) is True
        cond.traverse({})
        assert cond.is_open({}) is False

    def test_count_condition_traverse_increments(self):
        cond = CountCondition(limit=3)
        state = {}

        assert cond._counter == 0
        cond.traverse(state)
        assert cond._counter == 1
        cond.traverse(state)
        assert cond._counter == 2

    def test_count_condition_multiple_traversals(self):
        cond = CountCondition(limit=2)
        state = {}

        # First traversal - should be open
        assert cond.is_open(state) is True
        cond.traverse(state)

        # Second traversal - should be open
        assert cond.is_open(state) is True
        cond.traverse(state)

        # Third attempt - should be closed
        assert cond.is_open(state) is False

    def test_count_condition_from_dict(self):
        data = {"limit": 5}
        cond = CountCondition.from_dict(data)
        assert cond.limit == 5

    def test_count_condition_from_dict_default(self):
        data = {}
        cond = CountCondition.from_dict(data)
        assert cond.limit == 1


class TestRegexCondition:
    """Test RegexCondition for pattern matching."""

    def test_regex_condition_creation(self):
        cond = RegexCondition(regex=r"\d+")
        assert cond.regex == r"\d+"
        assert cond.matchtype == "search"

    def test_regex_condition_search_match(self):
        cond = RegexCondition(regex=r"\d+", matchtype="search")
        assert cond.is_open({"current_input": "has 123 numbers"}) is True
        assert cond.is_open({"current_input": "no numbers"}) is False

    def test_regex_condition_match_type(self):
        cond = RegexCondition(regex=r"^\d+", matchtype="match")
        assert cond.is_open({"current_input": "123 at start"}) is True
        assert cond.is_open({"current_input": "not at start 123"}) is False

    def test_regex_condition_fullmatch_type(self):
        cond = RegexCondition(regex=r"\d+", matchtype="fullmatch")
        assert cond.is_open({"current_input": "123"}) is True
        assert cond.is_open({"current_input": "123 extra"}) is False

    def test_regex_condition_no_current_input(self):
        cond = RegexCondition(regex=r"\d+")
        assert cond.is_open({}) is False

    def test_regex_condition_invalid_match_type_raises(self):
        with pytest.raises(ValueError, match="Invalid matchtype"):
            RegexCondition(regex=r"\d+", matchtype="invalid")

    def test_regex_condition_from_dict(self):
        data = {"regex": r"[A-Z]+"}
        cond = RegexCondition.from_dict(data)
        assert cond.regex == r"[A-Z]+"

    def test_regex_condition_complex_pattern(self):
        """Test with a complex regex pattern."""
        cond = RegexCondition(regex=r"error: (\w+)", matchtype="search")
        assert cond.is_open({"current_input": "error: not_found"}) is True
        assert cond.is_open({"current_input": "success: ok"}) is False


class TestConditionManager:
    """Test ConditionManager for managing conditions."""

    @patch("pithos.conditions.ConfigManager")
    def test_condition_manager_creation(self, mock_config):
        manager = ConditionManager(mock_config)
        assert manager.config_manager is mock_config
        assert isinstance(manager.static_conditions, dict)

    @patch("pithos.conditions.ConfigManager")
    def test_condition_manager_loads_static_conditions(self, mock_config):
        manager = ConditionManager(mock_config)
        # Should have loaded built-in conditions
        assert "Condition" in manager.static_conditions
        assert "CountCondition" in manager.static_conditions
        assert "RegexCondition" in manager.static_conditions
        assert "AlwaysCondition" in manager.static_conditions

    @patch("pithos.conditions.ConfigManager")
    def test_get_registered_condition_always(self, mock_config):
        mock_config.get_config.return_value = None
        manager = ConditionManager(mock_config)

        cond = manager.get_registered_condition("AlwaysCondition")
        assert cond is AlwaysCondition

    @patch("pithos.conditions.ConfigManager")
    def test_get_registered_condition_count(self, mock_config):
        mock_config.get_config.return_value = None
        manager = ConditionManager(mock_config)

        cond = manager.get_registered_condition("CountCondition", limit=5)
        assert isinstance(cond, CountCondition)
        assert cond.limit == 5

    @patch("pithos.conditions.ConfigManager")
    def test_get_registered_condition_from_config(self, mock_config):
        config_data = {"type": "CountCondition", "limit": 10}
        mock_config.get_config.return_value = config_data
        manager = ConditionManager(mock_config)

        cond = manager.get_registered_condition("test_condition")
        assert isinstance(cond, CountCondition)
        assert cond.limit == 10

    @patch("pithos.conditions.ConfigManager")
    def test_get_registered_condition_not_found_raises(self, mock_config):
        mock_config.get_config.return_value = None
        mock_config.list_registered_conditions.return_value = []
        manager = ConditionManager(mock_config)

        with pytest.raises(FileNotFoundError, match="not a registered condition"):
            manager.get_registered_condition("NonExistentCondition")


class TestConditionIntegration:
    """Integration tests for conditions."""

    def test_condition_chain(self):
        """Test multiple conditions in sequence."""
        count_cond = CountCondition(limit=2)
        regex_cond = RegexCondition(regex=r"^ready")  # Match "ready" at start

        state1 = {"current_input": "ready to go"}
        state2 = {"current_input": "not ready"}

        # Count condition allows first two
        assert count_cond.is_open(state1) is True
        count_cond.traverse(state1)
        assert count_cond.is_open(state1) is True
        count_cond.traverse(state1)
        assert count_cond.is_open(state1) is False

        # Regex condition checks content
        assert regex_cond.is_open(state1) is True
        assert regex_cond.is_open(state2) is False

    def test_custom_condition_logic(self):
        """Test custom condition with complex logic."""

        def complex_check(state):
            value = state.get("value", 0)
            flag = state.get("flag", False)
            return value > 10 and flag

        cond = Condition(condition=complex_check)

        assert cond.is_open({"value": 15, "flag": True}) is True
        assert cond.is_open({"value": 15, "flag": False}) is False
        assert cond.is_open({"value": 5, "flag": True}) is False

    def test_condition_serialization_roundtrip(self):
        """Test that conditions can be serialized and restored."""
        original = CountCondition(limit=5)
        data = original.to_dict()

        # Simulate saving/loading
        restored = CountCondition.from_dict(data)
        assert restored.limit == original.limit
        assert restored._counter == 0
