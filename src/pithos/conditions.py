"""Conditional edges for flowchart transitions."""

import logging
import yaml
import argparse
from typing import Callable, Any, Optional
import re
import inspect
from pathlib import Path
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Edge:
    """Base class for flowchart edges."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def is_open(self, state: dict[str, Any]) -> bool:
        """Check if the edge can be traversed."""
        return self.enabled

    def traverse(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform any side effects when traversing the edge."""
        return state


class Condition(Edge):
    """Conditional edge for flowchart transitions."""

    def __init__(
        self,
        condition: Callable[[dict[str, Any]], bool],
        *args: Any,
        registered_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(True)
        self.condition = condition
        self.registered = registered_name is not None
        self.registered_name = registered_name or self.__class__.__name__

    def is_open(self, state: dict[str, Any]) -> bool:
        """Evaluate the condition against the current state."""
        return self.condition(state)

    @classmethod
    def from_dict(cls, condition_dict: dict[str, Any]) -> "Condition":
        """Create a condition from a dictionary.

        Args:
            condition_dict: Dictionary containing condition configuration.

        Raises:
            ValueError: If condition_dict is None or empty.
            SyntaxError: If condition code cannot be evaluated.
        """
        if not condition_dict:
            raise ValueError("condition_dict cannot be None or empty")

        condition_code = condition_dict.get("condition", "lambda x: True")
        try:
            condition = eval(condition_code)
        except Exception as e:
            line_number_match = re.search(r"line (\d+)", str(e))
            line_number = line_number_match.group(1) if line_number_match else None
            if line_number is not None:
                line_number = int(line_number) - 1
                condition_code = condition_code.split("\n")
                condition_code[line_number] = (
                    f"{condition_code[line_number]} <---- Error: {e}"
                )
                condition_code = "\n".join(condition_code)
            raise SyntaxError(f"Error evaluating condition:\n{condition_code}")
        return Condition(condition=condition)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Condition":
        """Load condition from YAML file.

        Args:
            yaml_path: Path to YAML file.

        Raises:
            FileNotFoundError: If yaml_path doesn't exist.
            ValueError: If yaml_path is empty.
        """
        if not yaml_path or not yaml_path.strip():
            raise ValueError("yaml_path cannot be empty")
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    @classmethod
    def from_registered(
        cls, config_name: str, config_manager: ConfigManager
    ) -> "Condition":
        """Load condition from registered configuration."""
        return cls.from_dict(config_manager.get_config(config_name, "conditions"))

    def register(
        self, config_manager: ConfigManager, registered_name: Optional[str] = None
    ) -> None:
        """Register this condition."""
        self.registered = True
        self.registered_name = registered_name or self.registered_name
        config_manager.register_config(
            self.to_dict(), self.registered_name, "conditions"
        )

    def __str__(self) -> str:
        """String representation of the condition."""
        return f"{self.registered_name}({self.__dict__})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize condition to dictionary."""
        d: dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if key == "condition":
                if self.condition is not None and not self.registered:
                    source = inspect.getsource(self.condition)
                    d["condition"] = '"""' + source + '"""'
            elif key.startswith("registered"):
                continue
            elif key.startswith("_"):
                continue
            elif val is not None:
                d[key] = self.__dict__[key]
        d["type"] = self.registered_name
        return d

    def to_yaml(self, yaml_path: str) -> None:
        """Serialize condition to YAML file.

        Args:
            yaml_path: Path where YAML file will be written.

        Raises:
            ValueError: If yaml_path is empty.
        """
        if not yaml_path or not yaml_path.strip():
            raise ValueError("yaml_path cannot be empty")

        # Ensure parent directory exists
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(yaml_path, "w") as file:
            yaml.safe_dump(data, file)


AlwaysCondition = Condition(condition=lambda x: True, registered_name="AlwaysCondition")


class CountCondition(Condition):
    """Condition that is open for a limited number of traversals."""

    def __init__(self, limit: int = 1, *args: Any, **kwargs: Any) -> None:
        """Initialize CountCondition.

        Args:
            limit: Number of times this condition can be traversed.

        Raises:
            ValueError: If limit is less than 1.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        super().__init__(
            None,  # type: ignore
            registered_name=kwargs.get("registered_name", "CountCondition"),
        )
        self.limit = limit
        self._counter = 0

    def is_open(self, state: dict[str, Any]) -> bool:
        """Check if traversal count is below limit."""
        return self._counter < self.limit

    def traverse(self, state: dict[str, Any]) -> dict[str, Any]:
        """Increment counter when traversed."""
        self._counter += 1
        return state

    @classmethod
    def from_dict(cls, condition_dict: dict[str, Any]) -> "CountCondition":
        """Create CountCondition from dictionary."""
        return CountCondition(condition_dict.get("limit", 1))


class RegexCondition(Condition):
    """Condition based on regex matching against current_input in state."""

    def __init__(
        self, regex: str = "", matchtype: str = "search", *args: Any, **kwargs: Any
    ) -> None:
        """Initialize RegexCondition.

        Args:
            regex: Regex pattern to use for matching.
            matchtype: Type of matching (search, match, or fullmatch).

        Raises:
            ValueError: If regex is empty or matchtype is invalid.
        """
        if not regex:
            raise ValueError("regex cannot be empty")
        if matchtype not in ["search", "match", "fullmatch"]:
            raise ValueError(
                f"Invalid matchtype: {matchtype}. Must be 'search', 'match', or 'fullmatch'"
            )

        super().__init__(
            None,  # type: ignore
            registered_name=kwargs.get("registered_name", "RegexCondition"),
        )
        self.matchtype = matchtype
        self.regex = regex
        self._compiled_regex = re.compile(regex)

    def is_open(self, state: dict[str, Any]) -> bool:
        """Check if regex matches current_input in state."""
        text_input = state.get("current_input", "")
        if self.matchtype == "search":
            return self._compiled_regex.search(text_input) is not None
        elif self.matchtype == "match":
            return self._compiled_regex.match(text_input) is not None
        elif self.matchtype == "fullmatch":
            return self._compiled_regex.fullmatch(text_input) is not None
        else:
            raise ValueError(f"Invalid match type: {self.matchtype}")

    @classmethod
    def from_dict(cls, condition_dict: dict[str, Any]) -> "RegexCondition":
        """Create RegexCondition from dictionary.

        Raises:
            KeyError: If 'regex' key is missing from condition_dict.
        """
        if "regex" not in condition_dict:
            raise KeyError("Missing required 'regex' key in condition_dict")
        return RegexCondition(condition_dict["regex"])


class ConditionManager:
    """Manages built-in and registered conditions."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self.static_conditions: dict[str, Any] = {}
        self.load_conditions()

    def load_conditions(self) -> None:
        """Load all static conditions from module globals."""
        for name, cls in globals().items():
            if (isinstance(cls, type) and issubclass(cls, Condition)) or isinstance(
                cls, Condition
            ):
                self.static_conditions[name] = cls

    def is_registered(self, name: str) -> bool:
        return (
            name in self.static_conditions
            or name in self.config_manager.get_registered_condition_names()
        )

    def get_location(self, name: str) -> str:
        loc = self.config_manager.get_config_file(name, "conditions")
        if loc is None:
            return "Built-in"
        return loc

    def get_registered_condition(
        self, condition_name: str, *args, **kwargs
    ) -> Condition:
        """Get a registered condition by name.

        Args:
            condition_name: Name of the condition to retrieve.

        Raises:
            ValueError: If condition_name is empty.
            NotImplementedError: If condition type is not found.
        """
        if not condition_name or not condition_name.strip():
            raise ValueError("condition_name cannot be empty")

        # Check static conditions first (takes priority over config)
        if condition_name in self.static_conditions:
            cond = self.static_conditions[condition_name]
            # If it's an instance (like AlwaysCondition), return it directly
            if isinstance(cond, Condition):
                return cond
            # If it's a class, instantiate it
            return cond(*args, **kwargs)

        # Then check config-registered conditions
        data = self.config_manager.get_config(condition_name, "conditions")
        if data is not None:
            dt = data.get("type", "AlwaysCondition")
            if dt in self.static_conditions:
                return self.static_conditions[dt].from_dict(data)

            raise NotImplementedError(
                f"Error in condition {condition_name}, no static condition named: {dt}"
            )

        logger.debug(
            "Registered conditions: %s",
            list(self.config_manager.get_registered_condition_names()),
        )
        logger.debug("Static conditions: %s", list(self.static_conditions.keys()))
        raise FileNotFoundError(
            f"Condition {condition_name} is not a registered condition"
        )


def test_condition_loop(condition: Condition) -> None:
    """Interactive testing loop for conditions."""
    try:
        while True:
            user_input = input("Enter a string to test (CTRL+C to quit): ")
            result = condition({"current_input": user_input})
            print(f"Condition result: {result}")
    except KeyboardInterrupt:
        print("\nExiting...")


def main() -> None:
    """CLI entry point for condition management."""
    parser = argparse.ArgumentParser(description="Condition Manager CLI")
    subparsers = parser.add_subparsers(dest="command")

    register_parser = subparsers.add_parser("register", help="Register a new condition")
    register_parser.add_argument("condition", type=str, help="Path to the config file")
    register_parser.add_argument(
        "--name", type=str, help="New registered name for condition"
    )

    show_parser = subparsers.add_parser("show", help="Show registered conditions")
    show_parser.add_argument(
        "condition",
        type=str,
        help="Path to the config file, or registered name of condition",
    )

    test_parser = subparsers.add_parser(
        "test", help="test a condition against a string"
    )
    test_parser.add_argument(
        "condition", type=str, help="Condition name/config file to test"
    )

    subparsers.add_parser("list", help="List all registered conditions")

    args, param_args = parser.parse_known_args()
    param_dict = {}
    for i in range(0, len(param_args), 2):
        if param_args[i].startswith("--"):
            key = param_args[i][2:]
            value = param_args[i + 1] if i + 1 < len(param_args) else None
            try:
                param_dict[key] = int(value)
            except ValueError:
                param_dict[key] = value

    config_manager = ConfigManager()
    condition_manager = ConditionManager(config_manager)

    condition = None
    if "condition" in args:
        cond_loc = Path(args.condition)
        if cond_loc.exists():
            condition = Condition.from_yaml(args.condition)
            print(f"Loaded condition from YAML file {cond_loc}.")
        else:
            condition = condition_manager.get_registered_condition(
                args.condition, **param_dict
            )
            print(
                f"Loaded condition from registered config: {condition_manager.get_location(args.condition)}."
            )

    if args.command == "register":
        name = args.name or cond_loc.stem
        condition.register(config_manager, name)
        print(f"Condition registered as {name}")
    elif args.command == "show":
        print(condition)
    elif args.command == "test":
        test_condition_loop(condition)
    elif args.command == "list":
        print("Registered conditions:")
        for name in config_manager.get_registered_condition_names():
            print(name)
        print("Static (Built-in) conditions:")
        for name in condition_manager.static_conditions.keys():
            print(name)
    else:
        parser.print_help()
