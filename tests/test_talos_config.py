"""Tests for talos.config — dataclasses, YAML round-trip, wizard, build_agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from talos.config import (
    AgentConfig,
    DEFAULT_WAKE_WORD,
    MemoryConfig,
    TalosConfig,
    TelegramConfig,
    ToolsConfig,
    VoiceConfig,
    _ModeOverrideConfigManager,
    build_agent,
    ensure_config,
    load_config,
    save_config,
)


def test_defaults() -> None:
    cfg = TalosConfig()
    assert cfg.agent.model == "glm-4.7-flash"
    assert cfg.agent.tools == ToolsConfig()
    assert cfg.agent.memory == MemoryConfig()
    assert cfg.agent.tools.enabled is False
    assert cfg.agent.memory.enabled is False
    assert cfg.voice.device == "cuda"
    assert cfg.voice.wake_word == DEFAULT_WAKE_WORD
    assert cfg.telegram.bot_token == ""


def test_tools_config_defaults() -> None:
    tc = ToolsConfig()
    assert tc.enabled is False
    assert tc.mode == "include"
    assert tc.auto_loop is False
    assert tc.max_iterations == 5


def test_memory_config_defaults() -> None:
    mc = MemoryConfig()
    assert mc.enabled is False
    assert mc.persist_directory is None
    assert mc.compaction is False
    assert mc.compaction_threshold == 20
    assert mc.recall is False
    assert mc.history is False
    assert mc.tag_suggestions_model is None


def test_to_dict_from_dict_roundtrip() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            system_prompt="sp",
            temperature=0.2,
            tools=ToolsConfig(
                enabled=True, mode="all", auto_loop=True, max_iterations=3
            ),
            memory=MemoryConfig(
                enabled=True,
                compaction=True,
                compaction_threshold=15,
                recall=True,
                history=True,
                tag_suggestions_model="llama3",
            ),
        ),
        voice=VoiceConfig(device="cuda"),
        telegram=TelegramConfig(bot_token="abc"),
    )
    restored = TalosConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_agent_config_from_dict_handles_missing_nested_sections() -> None:
    ac = AgentConfig.from_dict({"model": "x"})
    assert ac.model == "x"
    assert ac.tools == ToolsConfig()
    assert ac.memory == MemoryConfig()


def test_agent_config_from_dict_nested_dicts() -> None:
    ac = AgentConfig.from_dict(
        {
            "model": "m",
            "tools": {"enabled": True, "mode": "exclude"},
            "memory": {"enabled": True, "recall": True},
        }
    )
    assert ac.tools == ToolsConfig(enabled=True, mode="exclude")
    assert ac.memory == MemoryConfig(enabled=True, recall=True)


def test_from_dict_handles_missing_sections() -> None:
    cfg = TalosConfig.from_dict({})
    assert cfg == TalosConfig()
    cfg2 = TalosConfig.from_dict({"agent": {"model": "x"}})
    assert cfg2.agent.model == "x"
    assert cfg2.voice == VoiceConfig()


def test_save_and_load(tmp_path: Path) -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="custom-model",
            tools=ToolsConfig(enabled=True),
            memory=MemoryConfig(enabled=False),
        ),
        telegram=TelegramConfig(bot_token="t0k3n"),
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert path.exists()
    with path.open() as f:
        raw = yaml.safe_load(f)
    assert raw["agent"]["model"] == "custom-model"
    assert raw["agent"]["tools"]["enabled"] is True
    assert raw["agent"]["memory"]["enabled"] is False
    assert raw["telegram"]["bot_token"] == "t0k3n"

    restored = load_config(path)
    assert restored == cfg


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dirs" / "config.yaml"
    save_config(TalosConfig(), path)
    assert path.exists()


def test_ensure_config_loads_existing(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    cfg = TalosConfig(agent=AgentConfig(model="existing"))
    save_config(cfg, path)

    with patch("talos.config.run_wizard") as wizard:
        loaded, returned_path = ensure_config(path)
    wizard.assert_not_called()
    assert returned_path == path
    assert loaded.agent.model == "existing"


def test_ensure_config_runs_wizard_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    fake_cfg = TalosConfig(agent=AgentConfig(model="wizard-model"))

    with patch("talos.config.run_wizard", return_value=fake_cfg) as wizard:
        loaded, returned_path = ensure_config(path)
    wizard.assert_called_once()
    assert returned_path == path
    assert path.exists()
    assert loaded.agent.model == "wizard-model"


def test_ensure_config_force_wizard(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(TalosConfig(agent=AgentConfig(model="old")), path)
    fake = TalosConfig(agent=AgentConfig(model="new"))

    with patch("talos.config.run_wizard", return_value=fake) as wizard:
        loaded, _ = ensure_config(path, force_wizard=True)
    wizard.assert_called_once()
    assert loaded.agent.model == "new"


def test_build_agent_returns_configured_ollama_agent() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="my-model",
            system_prompt="custom prompt",
            temperature=0.3,
            tools=ToolsConfig(enabled=False),
            memory=MemoryConfig(enabled=False),
        ),
    )
    agent = build_agent(cfg)
    assert agent.default_model == "my-model"
    assert agent.agent_name == "talos"
    assert agent.default_system_prompt == "custom prompt"
    assert agent.temperature == 0.3
    assert agent.tools_enabled is False
    assert agent.memory_enabled is False


def test_build_agent_enables_tools_when_requested() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            tools=ToolsConfig(enabled=True, auto_loop=True, max_iterations=2),
            memory=MemoryConfig(enabled=False),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_tools") as enable_tools, patch(
        "talos.config.OllamaAgent.enable_memory"
    ) as enable_memory:
        build_agent(cfg)
    enable_tools.assert_called_once()
    kwargs = enable_tools.call_args.kwargs
    assert kwargs["auto_loop"] is True
    assert kwargs["max_iterations"] == 2
    enable_memory.assert_not_called()


def test_build_agent_enables_memory_when_requested() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            tools=ToolsConfig(enabled=False),
            memory=MemoryConfig(enabled=True),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_tools") as enable_tools, patch(
        "talos.config.OllamaAgent.enable_memory"
    ) as enable_memory:
        build_agent(cfg)
    enable_tools.assert_not_called()
    enable_memory.assert_called_once()


def test_build_agent_tools_mode_default_uses_plain_config_manager() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            tools=ToolsConfig(enabled=True, mode="include"),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_tools") as enable_tools, patch(
        "talos.config.ConfigManager"
    ) as plain_cm, patch("talos.config._ModeOverrideConfigManager") as override_cm:
        build_agent(cfg)
    plain_cm.assert_called_once()
    override_cm.assert_not_called()
    enable_tools.assert_called_once()


def test_build_agent_tools_mode_override_uses_override_config_manager() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            tools=ToolsConfig(enabled=True, mode="all"),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_tools") as enable_tools, patch(
        "talos.config._ModeOverrideConfigManager"
    ) as override_cm:
        build_agent(cfg)
    override_cm.assert_called_once_with(tool_mode_override="all")
    enable_tools.assert_called_once()


def test_build_agent_memory_full_features() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            memory=MemoryConfig(
                enabled=True,
                persist_directory="/tmp/talos-mem",
                compaction=True,
                compaction_threshold=12,
                recall=True,
                history=True,
                tag_suggestions_model="llama3",
            ),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_memory") as em, patch(
        "talos.config.OllamaAgent.enable_compaction"
    ) as ec, patch("talos.config.OllamaAgent.enable_recall") as er, patch(
        "talos.config.OllamaAgent.enable_history"
    ) as eh, patch(
        "talos.config.OllamaAgent.enable_tag_suggestions"
    ) as ets:
        build_agent(cfg)
    em.assert_called_once()
    assert em.call_args.kwargs["persist_directory"] == "/tmp/talos-mem"
    ec.assert_called_once()
    assert ec.call_args.args[0].threshold == 12
    er.assert_called_once()
    eh.assert_called_once()
    assert eh.call_args.kwargs["persist_directory"] == "/tmp/talos-mem"
    ets.assert_called_once_with(model="llama3")


def test_build_agent_memory_compaction_only() -> None:
    cfg = TalosConfig(
        agent=AgentConfig(
            model="m",
            memory=MemoryConfig(enabled=True, compaction=True),
        ),
    )
    with patch("talos.config.OllamaAgent.enable_memory"), patch(
        "talos.config.OllamaAgent.enable_compaction"
    ) as ec, patch("talos.config.OllamaAgent.enable_recall") as er, patch(
        "talos.config.OllamaAgent.enable_history"
    ) as eh, patch(
        "talos.config.OllamaAgent.enable_tag_suggestions"
    ) as ets:
        build_agent(cfg)
    ec.assert_called_once()
    er.assert_not_called()
    eh.assert_not_called()
    ets.assert_not_called()


def test_mode_override_config_manager_patches_tool_mode(tmp_path: Path) -> None:
    # Build a minimal config tree with a tool_config.yaml that has mode=include.
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    (tool_dir / "tool_config.yaml").write_text(
        "enabled: true\nmode: include\ninclude: [echo]\n"
    )
    cm = _ModeOverrideConfigManager(tool_mode_override="all", config_dir=str(tmp_path))
    cfg = cm.get_config("tool_config", "tools")
    assert cfg is not None
    assert cfg["mode"] == "all"
    # Other configs (different namespace/name) are not touched.
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "foo.yaml").write_text("mode: include\n")
    cm2 = _ModeOverrideConfigManager(tool_mode_override="all", config_dir=str(tmp_path))
    other = cm2.get_config("foo", "agents")
    assert other is not None
    assert other["mode"] == "include"
