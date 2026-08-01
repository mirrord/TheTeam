"""Tests for talos.tools_cmd — enable, disable, list, list-all."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from talos.config import AgentConfig, TalosConfig, ToolsConfig, save_config, load_config
from talos.tools_cmd import (
    VIRTUAL_TOOLS,
    cmd_disable,
    cmd_enable,
    cmd_list,
    cmd_list_all,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**tools_kwargs) -> TalosConfig:
    return TalosConfig(agent=AgentConfig(tools=ToolsConfig(**tools_kwargs)))


def _save_and_reload(cfg: TalosConfig, path: Path) -> TalosConfig:
    save_config(cfg, path)
    return load_config(path)


# ---------------------------------------------------------------------------
# cmd_enable
# ---------------------------------------------------------------------------


def test_enable_adds_to_allow(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("python", cfg, path)
    reloaded = load_config(path)
    assert "python" in reloaded.agent.tools.allow


def test_enable_is_idempotent(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, allow=["python"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("python", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.allow.count("python") == 1


def test_enable_removes_from_deny(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, deny=["python"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("python", cfg, path)
    reloaded = load_config(path)
    assert "python" in reloaded.agent.tools.allow
    assert "python" not in reloaded.agent.tools.deny


def test_enable_web_research_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("web-research", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.web_research == {"enabled": True}
    # Should not appear in allow list.
    assert "web-research" not in reloaded.agent.tools.allow


def test_enable_web_research_alias(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("web_research", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.web_research == {"enabled": True}


def test_enable_prompt2image_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("prompt2image", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.prompt2image == {"enabled": True}


def test_enable_flowcharts_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, flowcharts={"enabled": False, "timeout": 60})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("flowcharts", cfg, path)
    reloaded = load_config(path)
    # Must set enabled=True while preserving other keys.
    assert reloaded.agent.tools.flowcharts["enabled"] is True
    assert reloaded.agent.tools.flowcharts.get("timeout") == 60


def test_enable_flowchart_alias(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("flowchart", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.flowcharts == {"enabled": True}


def test_enable_craft_write_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("craft-write", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.craft_writing == {"enabled": True}
    # Should not appear in allow list.
    assert "craft-write" not in reloaded.agent.tools.allow


def test_enable_craft_writing_alias(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("craft_writing", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.craft_writing == {"enabled": True}


def test_enable_research_news_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("research-news", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.news_research == {"enabled": True}
    # Should not appear in allow list.
    assert "research-news" not in reloaded.agent.tools.allow


def test_enable_news_research_alias(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_enable("news_research", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.news_research == {"enabled": True}


# ---------------------------------------------------------------------------
# cmd_disable
# ---------------------------------------------------------------------------


def test_disable_adds_to_deny(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("curl", cfg, path)
    reloaded = load_config(path)
    assert "curl" in reloaded.agent.tools.deny


def test_disable_is_idempotent(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, deny=["curl"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("curl", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.deny.count("curl") == 1


def test_disable_removes_from_allow(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, allow=["curl"])
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("curl", cfg, path)
    reloaded = load_config(path)
    assert "curl" not in reloaded.agent.tools.allow
    assert "curl" in reloaded.agent.tools.deny


def test_disable_web_research_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, web_research={"enabled": True})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("web-research", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.web_research == {"enabled": False}
    assert "web-research" not in reloaded.agent.tools.deny


def test_disable_prompt2image_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, prompt2image={"enabled": True})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("prompt2image", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.prompt2image == {"enabled": False}


def test_disable_flowcharts_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, flowcharts={"enabled": True, "max_steps": 50})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("flowcharts", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.flowcharts["enabled"] is False
    assert reloaded.agent.tools.flowcharts.get("max_steps") == 50


def test_disable_craft_write_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, craft_writing={"enabled": True})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("craft-write", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.craft_writing == {"enabled": False}
    assert "craft-write" not in reloaded.agent.tools.deny


def test_craft_write_in_virtual_tools_map() -> None:
    assert VIRTUAL_TOOLS["craft-write"] == "craft_writing"
    assert VIRTUAL_TOOLS["craft_writing"] == "craft_writing"


def test_disable_research_news_virtual(tmp_path: Path) -> None:
    cfg = _make_config(enabled=True, news_research={"enabled": True})
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    cmd_disable("research-news", cfg, path)
    reloaded = load_config(path)
    assert reloaded.agent.tools.news_research == {"enabled": False}
    assert "research-news" not in reloaded.agent.tools.deny


def test_research_news_in_virtual_tools_map() -> None:
    assert VIRTUAL_TOOLS["research-news"] == "news_research"
    assert VIRTUAL_TOOLS["news-research"] == "news_research"
    assert VIRTUAL_TOOLS["news_research"] == "news_research"


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


def test_list_when_tools_globally_disabled(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=False)
    path = tmp_path / "config.yaml"
    cmd_list(cfg, path)
    captured = capsys.readouterr()
    assert "disabled" in captured.out.lower()


def test_list_with_registry_build_error(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    with patch(
        "talos.tools_cmd._build_registry", side_effect=RuntimeError("no config")
    ):
        cmd_list(cfg, path)
    captured = capsys.readouterr()
    assert "could not build" in captured.out.lower()


def test_list_empty_registry(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    mock_plain_cm = MagicMock()
    mock_plain_cm.get_registered_flowchart_names.return_value = []
    with patch("talos.tools_cmd._build_registry", return_value=(mock_registry, {})):
        with patch("talos.tools_cmd._try_rich", return_value=None):
            with patch("talos.tools_cmd.ConfigManager", return_value=mock_plain_cm):
                cmd_list(cfg, path)
    captured = capsys.readouterr()
    # No CLI tools → shows the "no CLI tools available" line
    assert "no cli tools" in captured.out.lower()


def test_list_shows_tools(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    mock_meta = MagicMock()
    mock_meta.description = "Python interpreter"
    mock_meta.tool_type = "cli"
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = ["python"]
    mock_registry.get_tool.return_value = mock_meta
    mock_registry.requires_confirmation.return_value = False
    mock_plain_cm = MagicMock()
    mock_plain_cm.get_registered_flowchart_names.return_value = []
    with patch("talos.tools_cmd._build_registry", return_value=(mock_registry, {})):
        with patch("talos.tools_cmd._try_rich", return_value=None):
            with patch("talos.tools_cmd.ConfigManager", return_value=mock_plain_cm):
                cmd_list(cfg, path)
    captured = capsys.readouterr()
    assert "python" in captured.out
    assert "Python interpreter" in captured.out


def test_list_shows_flowcharts_even_when_disabled(tmp_path: Path, capsys) -> None:
    """Configured flowcharts appear in list even when the flowchart tool is disabled."""
    cfg = _make_config(enabled=True, flowcharts={"enabled": False})
    path = tmp_path / "config.yaml"
    # Registry has NO flowchart entries (tool disabled).
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    # But plain ConfigManager finds two flowcharts on disk.
    mock_plain_cm = MagicMock()
    mock_plain_cm.get_registered_flowchart_names.return_value = [
        "research",
        "summarize",
    ]
    tool_cfg = {"flowcharts": {"enabled": False}}
    with patch(
        "talos.tools_cmd._build_registry", return_value=(mock_registry, tool_cfg)
    ):
        with patch("talos.tools_cmd._try_rich", return_value=None):
            with patch("talos.tools_cmd.ConfigManager", return_value=mock_plain_cm):
                cmd_list(cfg, path)
    out = capsys.readouterr().out
    assert "research" in out
    assert "summarize" in out
    # Should indicate they're disabled / not active
    assert "disabled" in out.lower() or "inactive" in out.lower()


# ---------------------------------------------------------------------------
# cmd_list_all
# ---------------------------------------------------------------------------


def _minimal_tool_config(
    mode: str = "strict",
    include: list | None = None,
    exclude: list | None = None,
    confirm: list | None = None,
) -> dict:
    return {
        "mode": mode,
        "include": include or ["python", "curl"],
        "exclude": exclude or ["rm"],
        "confirm": confirm or ["bash"],
        "descriptions": {"python": "Python interp", "curl": "HTTP client"},
        "flowcharts": {"enabled": True},
        "web_research": {"enabled": False},
        "prompt2image": {"enabled": False},
        "craft_writing": {"enabled": True},
        "news_research": {"enabled": True},
    }


def test_list_all_shows_status_labels(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"

    mock_registry = MagicMock()
    mock_registry.config = _minimal_tool_config()

    with patch("talos.tools_cmd.ToolRegistry", return_value=mock_registry):
        with patch("talos.tools_cmd._make_config_manager", return_value=MagicMock()):
            with patch("talos.tools_cmd._try_rich", return_value=None):
                with patch("shutil.which", return_value="/usr/bin/python"):
                    cmd_list_all(cfg, path)

    out = capsys.readouterr().out
    # allowed tool should appear
    assert "python" in out
    assert "allowed" in out
    # blocked tool
    assert "rm" in out
    assert "blocked" in out
    # confirm tool
    assert "bash" in out
    assert "confirm" in out


def test_list_all_shows_not_found_for_unlisted_tool(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"

    tool_cfg = _minimal_tool_config(include=["python", "notinstalled"])
    mock_registry = MagicMock()
    mock_registry.config = tool_cfg

    def _which(name: str):
        return None if name == "notinstalled" else f"/usr/bin/{name}"

    with patch("talos.tools_cmd.ToolRegistry", return_value=mock_registry):
        with patch("talos.tools_cmd._make_config_manager", return_value=MagicMock()):
            with patch("talos.tools_cmd._try_rich", return_value=None):
                with patch("shutil.which", side_effect=_which):
                    cmd_list_all(cfg, path)

    out = capsys.readouterr().out
    assert "notinstalled" in out
    assert "not-found" in out


def test_list_all_virtual_tools_shown(tmp_path: Path, capsys) -> None:
    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"

    mock_registry = MagicMock()
    mock_registry.config = _minimal_tool_config()

    with patch("talos.tools_cmd.ToolRegistry", return_value=mock_registry):
        with patch("talos.tools_cmd._make_config_manager", return_value=MagicMock()):
            with patch("talos.tools_cmd._try_rich", return_value=None):
                with patch("shutil.which", return_value="/usr/bin/x"):
                    cmd_list_all(cfg, path)

    out = capsys.readouterr().out
    assert "web-research" in out
    assert "flowcharts" in out
    assert "prompt2image" in out
    assert "craft-write" in out
    assert "research-news" in out
    assert "enabled" in out
    assert "enabled" in out


# ---------------------------------------------------------------------------
# run() dispatcher
# ---------------------------------------------------------------------------


def test_run_no_config_file(tmp_path: Path, capsys) -> None:
    import argparse

    args = argparse.Namespace(tools_action="list")
    rc = run(args, tmp_path / "nonexistent.yaml")
    assert rc == 1
    captured = capsys.readouterr()
    assert "no talos config" in captured.out.lower()


def test_run_enable_dispatches(tmp_path: Path) -> None:
    import argparse

    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    args = argparse.Namespace(tools_action="enable", tool_name="python")
    with patch("talos.tools_cmd.cmd_enable") as mock_enable:
        rc = run(args, path)
    assert rc == 0
    mock_enable.assert_called_once()
    assert mock_enable.call_args.args[0] == "python"


def test_run_disable_dispatches(tmp_path: Path) -> None:
    import argparse

    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    args = argparse.Namespace(tools_action="disable", tool_name="rm")
    with patch("talos.tools_cmd.cmd_disable") as mock_disable:
        rc = run(args, path)
    assert rc == 0
    mock_disable.assert_called_once()
    assert mock_disable.call_args.args[0] == "rm"


def test_run_list_dispatches(tmp_path: Path) -> None:
    import argparse

    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    for action in ("list", "ls"):
        args = argparse.Namespace(tools_action=action)
        with patch("talos.tools_cmd.cmd_list") as mock_list:
            rc = run(args, path)
        assert rc == 0
        mock_list.assert_called_once()


def test_run_list_all_dispatches(tmp_path: Path) -> None:
    import argparse

    cfg = _make_config(enabled=True)
    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    args = argparse.Namespace(tools_action="list-all")
    with patch("talos.tools_cmd.cmd_list_all") as mock_list_all:
        rc = run(args, path)
    assert rc == 0
    mock_list_all.assert_called_once()


# ---------------------------------------------------------------------------
# VIRTUAL_TOOLS mapping completeness
# ---------------------------------------------------------------------------


def test_virtual_tools_mapping_covers_expected_keys() -> None:
    assert "web-research" in VIRTUAL_TOOLS
    assert "prompt2image" in VIRTUAL_TOOLS
    assert "flowcharts" in VIRTUAL_TOOLS
    assert "flowchart" in VIRTUAL_TOOLS
    assert "web_research" in VIRTUAL_TOOLS
    # All values must be valid ToolsConfig attribute names.
    from talos.config import ToolsConfig as TC
    import dataclasses

    tc_fields = {f.name for f in dataclasses.fields(TC)}
    for attr in VIRTUAL_TOOLS.values():
        assert attr in tc_fields, f"VIRTUAL_TOOLS maps to unknown field: {attr!r}"
