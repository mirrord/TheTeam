"""Configuration management for pithos agents, flowcharts, and conditions."""

import logging
import os
from pathlib import Path
from typing import Optional, Any, Iterator
import yaml

logger = logging.getLogger(__name__)

# Environment variable name for overriding the config directory.
CONFIG_DIR_ENV_VAR = "PITHOS_CONFIG_DIR"


class ConfigManager:
    """Manages configuration files organized by namespace directories."""

    def __init__(self, config_dir: Optional[str] = None) -> None:
        """Initialize ConfigManager.

        Resolution order for the config directory:
        1. *config_dir* argument — absolute paths are used as-is; relative paths
           are resolved against the current working directory.
        2. ``PITHOS_CONFIG_DIR`` environment variable (absolute or CWD-relative).
        3. ``<cwd>/configs`` as a built-in default.

        Args:
            config_dir: Path to the directory containing configuration files.
                If *None* (the default), the ``PITHOS_CONFIG_DIR`` environment
                variable is consulted, falling back to ``<cwd>/configs``.

        Raises:
            ValueError: If config_dir is an empty string.
        """
        if config_dir is not None and not config_dir.strip():
            raise ValueError("config_dir cannot be empty")

        resolved: Optional[str] = config_dir or os.environ.get(CONFIG_DIR_ENV_VAR)
        if resolved is not None:
            path = Path(resolved)
            self.config_dir = path if path.is_absolute() else Path.cwd() / path
        else:
            self.config_dir = Path.cwd() / "configs"

        self.configs: dict[str, dict[str, Path]] = {}
        self.load_configs()

    def load_configs(self) -> None:
        """Load all YAML configs from the config directory."""
        for file_path in self.config_dir.rglob("*.yaml"):
            folder_name = str(file_path.parent.relative_to(self.config_dir))
            config_name = file_path.stem
            if folder_name not in self.configs:
                self.configs[folder_name] = {}
            self.configs[folder_name][config_name] = file_path

    def get_config_file(
        self, config_name: str, namespace: Optional[str] = None
    ) -> Optional[Path]:
        """Get the path to a configuration file.

        Args:
            config_name: Name of the configuration.
            namespace: Namespace directory (e.g., 'agents', 'flowcharts').

        Returns:
            Path to the config file, or None if not found.
        """
        return self.configs.get(namespace, {}).get(config_name, None)

    def get_config(
        self, config_name: str, namespace: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Load and parse a configuration file.

        Args:
            config_name: Name of the configuration.
            namespace: Namespace directory.

        Returns:
            Parsed configuration dict, or None if not found.

        Raises:
            ValueError: If config_name is empty.
        """
        if not config_name or not config_name.strip():
            raise ValueError("config_name cannot be empty")
        if namespace is not None and not namespace.strip():
            raise ValueError("namespace cannot be empty when provided")

        config_file = self.get_config_file(config_name, namespace)
        if config_file:
            with config_file.open("r") as f:
                return yaml.safe_load(f)
        return None

    def get_registered_config_names(self, namespace: str) -> Iterator[str]:
        """Get all registered config names in a namespace.

        Args:
            namespace: Namespace to get config names from.

        Raises:
            ValueError: If namespace is empty.
        """
        if not namespace or not namespace.strip():
            raise ValueError("namespace cannot be empty")
        return self.configs.get(namespace, {}).keys()

    def get_registered_namespaces(self) -> Iterator[str]:
        """Get all registered namespace names."""
        return self.configs.keys()

    def get_registered_agent_names(self) -> Iterator[str]:
        """Get all registered agent names."""
        return self.configs.get("agents", {}).keys()

    def get_registered_condition_names(self) -> Iterator[str]:
        """Get all registered condition names."""
        return self.configs.get("conditions", {}).keys()

    def get_registered_flowchart_names(self) -> Iterator[str]:
        """Get all registered flowchart names."""
        return self.configs.get("flowcharts", {}).keys()

    def register_config(
        self, config: dict[str, Any], config_name: str, namespace: str
    ) -> None:
        """Register a new configuration.

        Args:
            config: Configuration dictionary to save.
            config_name: Name for the configuration.
            namespace: Namespace directory to save in.

        Raises:
            ValueError: If config is None/empty or names are empty.
        """
        if not config:
            raise ValueError("config cannot be None or empty")
        if not config_name or not config_name.strip():
            raise ValueError("config_name cannot be empty")
        if not namespace or not namespace.strip():
            raise ValueError("namespace cannot be empty")

        namespace_dir = self.config_dir / namespace
        if namespace not in self.configs:
            self.configs[namespace] = {}
            namespace_dir.mkdir(parents=True, exist_ok=True)
        self.configs[namespace][config_name] = namespace_dir / f"{config_name}.yaml"
        with self.configs[namespace][config_name].open("w") as f:
            yaml.safe_dump(config, f)
        logger.info("Registered new %s config as %s", namespace, config_name)


def main() -> None:
    """CLI entry point for config management."""
    cm = ConfigManager()
    print("Note: Built-ins not listed.")
    print("\nNamespaces:")
    print("\n\t".join(cm.get_registered_namespaces()))
    print("\nAgents:")
    print("\n\t".join(cm.get_registered_agent_names()))
    print("\nConditions:")
    print("\n\t".join(cm.get_registered_condition_names()))
    print("\nFlowcharts:")
    print("\n\t".join(cm.get_registered_flowchart_names()))
