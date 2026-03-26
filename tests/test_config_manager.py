"""Unit tests for config_manager module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from pithos.config_manager import ConfigManager, CONFIG_DIR_ENV_VAR


class TestConfigManager:
    """Test ConfigManager for configuration handling."""

    def test_config_manager_creation_with_custom_dir(self):
        """Test creating config manager with an explicit absolute directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()

            agents_dir = config_dir / "agents"
            agents_dir.mkdir()
            config_file = agents_dir / "test_agent.yaml"
            config_file.write_text(
                yaml.dump({"model": "glm-4.7-flash", "name": "test"})
            )

            cm = ConfigManager(config_dir=str(config_dir))
            assert cm.config_dir == config_dir
            assert cm.get_config("test_agent", "agents") is not None

    def test_empty_string_config_dir_raises(self):
        """Test that an empty string config_dir raises ValueError."""
        with pytest.raises(ValueError, match="config_dir cannot be empty"):
            ConfigManager(config_dir="")

    def test_whitespace_config_dir_raises(self):
        """Test that a whitespace-only config_dir raises ValueError."""
        with pytest.raises(ValueError, match="config_dir cannot be empty"):
            ConfigManager(config_dir="   ")

    def test_default_uses_cwd_configs(self, monkeypatch, tmp_path):
        """Test that the default (no args, no env var) resolves to <cwd>/configs."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
        cm = ConfigManager()
        assert cm.config_dir == tmp_path / "configs"

    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        """Test that PITHOS_CONFIG_DIR env var takes precedence over built-in default."""
        custom_dir = tmp_path / "custom_configs"
        custom_dir.mkdir()
        monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(custom_dir))
        cm = ConfigManager()
        assert cm.config_dir == custom_dir

    def test_explicit_arg_overrides_env_var(self, monkeypatch, tmp_path):
        """Test that an explicit config_dir arg takes precedence over the env var."""
        env_dir = tmp_path / "env_configs"
        env_dir.mkdir()
        arg_dir = tmp_path / "arg_configs"
        arg_dir.mkdir()
        monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))
        cm = ConfigManager(config_dir=str(arg_dir))
        assert cm.config_dir == arg_dir

    def test_relative_path_resolves_against_cwd(self, monkeypatch, tmp_path):
        """Test that a relative config_dir is resolved against the CWD."""
        monkeypatch.chdir(tmp_path)
        rel_name = "my_configs"
        (tmp_path / rel_name).mkdir()
        cm = ConfigManager(config_dir=rel_name)
        assert cm.config_dir == tmp_path / rel_name

    def test_get_config_returns_none_for_missing(self):
        """Test that get_config returns None for non-existent configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            result = cm.get_config("nonexistent", "agents")
            assert result is None

    def test_register_config_creates_namespace(self):
        """Test that registering config creates namespace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            config = {"model": "test", "name": "agent1"}

            cm.register_config(config, "agent1", "agents")

            # Check that file was created
            expected_file = Path(tmpdir) / "agents" / "agent1.yaml"
            assert expected_file.exists()

            # Check that content is correct
            with open(expected_file) as f:
                loaded = yaml.safe_load(f)
            assert loaded["model"] == "test"

    def test_register_config_overwrites_existing(self):
        """Test that registering overwrites existing config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            # Register first version
            config1 = {"value": "first"}
            cm.register_config(config1, "test", "configs")

            # Register second version
            config2 = {"value": "second"}
            cm.register_config(config2, "test", "configs")

            # Should have second version
            loaded = cm.get_config("test", "configs")
            assert loaded["value"] == "second"

    def test_get_config_file_returns_path(self):
        """Test that get_config_file returns path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)
            config = {"test": "data"}
            cm.register_config(config, "myconfig", "namespace")

            path = cm.get_config_file("myconfig", "namespace")
            assert path is not None
            assert isinstance(path, Path)
            assert path.exists()

    def test_get_registered_config_names(self):
        """Test getting all config names in a namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            # Register multiple configs
            cm.register_config({"a": 1}, "config1", "test")
            cm.register_config({"b": 2}, "config2", "test")
            cm.register_config({"c": 3}, "config3", "test")

            names = list(cm.get_registered_config_names("test"))
            assert "config1" in names
            assert "config2" in names
            assert "config3" in names

    def test_get_registered_namespaces(self):
        """Test getting all namespace names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            # Register configs in different namespaces
            cm.register_config({"a": 1}, "c1", "agents")
            cm.register_config({"b": 2}, "c2", "flowcharts")
            cm.register_config({"c": 3}, "c3", "conditions")

            namespaces = list(cm.get_registered_namespaces())
            assert "agents" in namespaces
            assert "flowcharts" in namespaces
            assert "conditions" in namespaces

    def test_get_registered_agent_names(self):
        """Test getting agent config names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            cm.register_config({"model": "m1"}, "agent1", "agents")
            cm.register_config({"model": "m2"}, "agent2", "agents")

            names = list(cm.get_registered_agent_names())
            assert "agent1" in names
            assert "agent2" in names

    def test_get_registered_condition_names(self):
        """Test getting condition config names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            cm.register_config({"type": "count"}, "cond1", "conditions")

            names = list(cm.get_registered_condition_names())
            assert "cond1" in names

    def test_get_registered_flowchart_names(self):
        """Test getting flowchart config names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            cm.register_config({"nodes": {}}, "flow1", "flowcharts")

            names = list(cm.get_registered_flowchart_names())
            assert "flow1" in names

    def test_load_configs_on_init(self):
        """Test that existing configs are loaded on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config structure before initializing manager
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()

            config_file = agents_dir / "existing.yaml"
            config_file.write_text(yaml.dump({"model": "test"}))

            # Initialize manager - should load existing configs
            cm = ConfigManager(config_dir=tmpdir)

            # Should be able to get the existing config
            config = cm.get_config("existing", "agents")
            assert config is not None
            assert config["model"] == "test"

    def test_register_config_with_nested_data(self):
        """Test registering config with nested structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            config = {
                "name": "agent",
                "model": "glm-4.7-flash",
                "contexts": {
                    "ctx1": {"system_prompt": "Test"},
                    "ctx2": {"system_prompt": "Test2"},
                },
                "inference": {"type": "flowchart", "steps": ["step1", "step2"]},
            }

            cm.register_config(config, "complex_agent", "agents")

            loaded = cm.get_config("complex_agent", "agents")
            assert loaded["name"] == "agent"
            assert "contexts" in loaded
            assert loaded["contexts"]["ctx1"]["system_prompt"] == "Test"
            assert loaded["inference"]["type"] == "flowchart"


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def test_full_workflow(self):
        """Test complete workflow of registering and loading configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(config_dir=tmpdir)

            # Register agent config
            agent_config = {
                "model": "glm-4.7-flash",
                "name": "test_agent",
                "system_prompt": "You are helpful",
            }
            cm.register_config(agent_config, "test_agent", "agents")

            # Register flowchart config
            flowchart_config = {
                "nodes": {"node1": {"type": "prompt", "prompt": "Test"}},
                "edges": [],
                "start_node": "node1",
            }
            cm.register_config(flowchart_config, "test_flow", "flowcharts")

            # Register condition config
            condition_config = {"type": "CountCondition", "limit": 5}
            cm.register_config(condition_config, "test_cond", "conditions")

            # Verify all can be retrieved
            assert cm.get_config("test_agent", "agents") is not None
            assert cm.get_config("test_flow", "flowcharts") is not None
            assert cm.get_config("test_cond", "conditions") is not None

            # Verify namespaces exist
            namespaces = list(cm.get_registered_namespaces())
            assert "agents" in namespaces
            assert "flowcharts" in namespaces
            assert "conditions" in namespaces

    def test_multiple_managers_same_directory(self):
        """Test that multiple managers can access same config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm1 = ConfigManager(config_dir=tmpdir)
            cm1.register_config({"value": "test"}, "shared", "namespace")

            # Create second manager pointing to same directory
            cm2 = ConfigManager(config_dir=tmpdir)

            # Second manager should see the config
            config = cm2.get_config("shared", "namespace")
            assert config is not None
            assert config["value"] == "test"
