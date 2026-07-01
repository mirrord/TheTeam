"""``talos tools`` subcommand — enable, disable, list and inspect tools.

Provides four actions:

* ``enable <name>``  — add a tool to the Talos-local allow list (or enable a
  virtual tool) by editing ``~/.talos/config.yaml``.  Does not touch the
  shared ``configs/tools/tool_config.yaml``.
* ``disable <name>`` — add a tool to the Talos-local deny list (or disable a
  virtual tool).
* ``list`` / ``ls``  — show every tool that would be available to the Talos
  agent given current config settings.
* ``list-all``       — show all tools from the config lists plus their
  discoverability status on this system.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

from pithos import ConfigManager
from pithos.tools.registry import ToolRegistry

from .config import (
    DEFAULT_TOOLS_MODE,
    TalosConfig,
    ToolsConfig,
    _ModeOverrideConfigManager,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# Virtual tool registry
# ---------------------------------------------------------------------------

# Maps the CLI name a user types to the ToolsConfig attribute that controls it.
VIRTUAL_TOOLS: dict[str, str] = {
    "web-research": "web_research",
    "web_research": "web_research",
    "text2image": "text2image",
    "flowcharts": "flowcharts",
    "flowchart": "flowcharts",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_config_manager(tc: ToolsConfig) -> ConfigManager:
    """Build the appropriate ConfigManager for the given ToolsConfig.

    Mirrors the selection logic in :func:`~talos.config.build_agent` —
    returns a :class:`~talos.config._ModeOverrideConfigManager` when any
    Talos-level override is active, otherwise a plain
    :class:`~pithos.ConfigManager`.
    """
    tool_overrides: dict[str, Any] = {}
    if tc.flowcharts is not None:
        tool_overrides["flowcharts"] = tc.flowcharts
    if tc.web_research is not None:
        tool_overrides["web_research"] = tc.web_research
    if tc.text2image is not None:
        tool_overrides["text2image"] = tc.text2image

    needs_override = (
        tc.mode != DEFAULT_TOOLS_MODE
        or bool(tool_overrides)
        or bool(tc.allow)
        or bool(tc.deny)
    )
    if not needs_override:
        return ConfigManager()

    kwargs: dict[str, Any] = {}
    # Only propagate a mode change when the user actually changed it from the
    # default ("include").  Passing "include" as an override would write an
    # unrecognised mode name into tool_config, causing CLIToolProvider to
    # block all tools silently.
    if tc.mode != DEFAULT_TOOLS_MODE:
        kwargs["tool_mode_override"] = tc.mode
    if tool_overrides:
        kwargs["tool_config_overrides"] = tool_overrides
    if tc.allow:
        kwargs["allow"] = tc.allow
    if tc.deny:
        kwargs["deny"] = tc.deny
    return _ModeOverrideConfigManager(**kwargs)


def _build_registry(tc: ToolsConfig, cm: ConfigManager):
    """Build a ToolRegistry populated with the same providers as the agent.

    Mirrors the provider-assembly logic in
    :meth:`pithos.OllamaAgent.enable_tools` without creating an agent.

    Returns:
        ``(ToolRegistry, raw_tool_config)`` tuple.
    """
    from pithos.tools.cli_provider import CLIToolProvider
    from pithos.tools.registry import ToolRegistry
    from pithos.tools.safety import CommandSafetyChecker

    # Load config via a bare registry (reads tool_config.yaml with overrides).
    _tmp = ToolRegistry(cm, providers=[])
    tool_config: dict[str, Any] = _tmp.config

    safety = CommandSafetyChecker(tool_config.get("safety", {}))
    cli = CLIToolProvider(
        config=tool_config,
        timeout=tool_config.get("timeout", 30),
        max_output_size=tool_config.get("max_output_size", 10000),
        safety_checker=safety,
    )
    providers = [cli]

    fc_config = tool_config.get("flowcharts", {})
    if fc_config.get("enabled", False):
        try:
            from pithos.tools.flowchart_tool import FlowchartToolExecutor

            providers.append(FlowchartToolExecutor(config_manager=cm))
        except Exception:
            pass

    wr_config = tool_config.get("web_research", {})
    if wr_config.get("enabled", False):
        try:
            from pithos.tools.web_researcher import (
                WEB_RESEARCH_AVAILABLE,
                WebResearcherToolExecutor,
            )

            if WEB_RESEARCH_AVAILABLE:
                providers.append(WebResearcherToolExecutor(config_manager=cm))
        except Exception:
            pass

    t2i_config = tool_config.get("text2image", {})
    if t2i_config.get("enabled", False):
        try:
            from pithos.tools.text2image import (
                TEXT2IMAGE_AVAILABLE,
                Text2ImageToolProvider,
            )

            if TEXT2IMAGE_AVAILABLE:
                providers.append(Text2ImageToolProvider(config_manager=cm))
        except Exception:
            pass

    return ToolRegistry(cm, providers=providers), tool_config


def _try_rich():
    """Return (Console, Table, Text, Style) or None if rich is not installed."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        return Console(), Table, Text
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_enable(tool_name: str, config: TalosConfig, config_path: Path) -> None:
    """Enable *tool_name* for the Talos agent.

    For virtual tools (web-research, text2image, flowcharts): sets the
    ``enabled: true`` flag in the relevant ``ToolsConfig`` dict field.
    For all other tools: appends to the ``allow`` list and removes any
    entry from the ``deny`` list.

    Changes are persisted to *config_path*.
    """
    tool_name = tool_name.strip()
    tc = config.agent.tools

    if tool_name in VIRTUAL_TOOLS:
        attr = VIRTUAL_TOOLS[tool_name]
        current: Optional[dict] = getattr(tc, attr)
        updated = dict(current) if current else {}
        updated["enabled"] = True
        setattr(tc, attr, updated)
        save_config(config, config_path)
        print(f"Virtual tool '{tool_name}' enabled (saved to {config_path}).")
        return

    # Regular tool.
    if tool_name not in tc.allow:
        tc.allow.append(tool_name)
    # Remove from deny if present.
    if tool_name in tc.deny:
        tc.deny.remove(tool_name)

    save_config(config, config_path)
    print(f"Tool '{tool_name}' added to allow list (saved to {config_path}).")


def cmd_disable(tool_name: str, config: TalosConfig, config_path: Path) -> None:
    """Disable *tool_name* for the Talos agent.

    For virtual tools: sets ``enabled: false``.  For regular tools:
    appends to the ``deny`` list and removes from ``allow``.

    Changes are persisted to *config_path*.
    """
    tool_name = tool_name.strip()
    tc = config.agent.tools

    if tool_name in VIRTUAL_TOOLS:
        attr = VIRTUAL_TOOLS[tool_name]
        current: Optional[dict] = getattr(tc, attr)
        updated = dict(current) if current else {}
        updated["enabled"] = False
        setattr(tc, attr, updated)
        save_config(config, config_path)
        print(f"Virtual tool '{tool_name}' disabled (saved to {config_path}).")
        return

    # Regular tool.
    if tool_name not in tc.deny:
        tc.deny.append(tool_name)
    # Remove from allow if present.
    if tool_name in tc.allow:
        tc.allow.remove(tool_name)

    save_config(config, config_path)
    print(f"Tool '{tool_name}' added to deny list (saved to {config_path}).")


def cmd_list(config: TalosConfig, config_path: Path) -> None:
    """Print tools that are currently available to the Talos agent.

    Displays CLI tools and virtual tools in separate sections, mirroring
    exactly what the agent's ToolRegistry contains at startup.  Configured
    flowcharts are always listed, even when the flowchart virtual tool is
    disabled in the Talos config, so users can see what workflows exist.
    """
    tc = config.agent.tools

    if not tc.enabled:
        print(
            "Tools are disabled for the Talos agent "
            "(set agent.tools.enabled: true in config to enable them)."
        )
        return

    cm = _make_config_manager(tc)
    try:
        registry, tool_config = _build_registry(tc, cm)
    except Exception as exc:
        print(f"Could not build tool registry: {exc}")
        return

    # --- Partition the registry by tool type --------------------------------
    cli_entries: list[tuple[str, Any]] = []
    flowchart_entries: list[tuple[str, Any]] = []
    virtual_entries: list[tuple[str, Any]] = []

    for name in registry.list_tools():
        meta = registry.get_tool(name)
        ttype = meta.tool_type if meta else "cli"
        if ttype == "flowchart":
            flowchart_entries.append((name, meta))
        elif ttype in ("web_research", "memory", "text2image"):
            virtual_entries.append((name, meta))
        else:
            cli_entries.append((name, meta))

    # --- Collect ALL configured flowcharts (even when tool is disabled) -----
    # Use a plain ConfigManager so the talos flowcharts.enabled override
    # doesn't hide flowchart YAML files that the user may want to see.
    try:
        plain_cm = ConfigManager()
        all_fc_names: list[str] = sorted(plain_cm.get_registered_flowchart_names())
    except Exception:
        all_fc_names = []

    active_fc_names: set[str] = {
        name.removeprefix("flowchart:")
        for name, _ in flowchart_entries
        if name.startswith("flowchart:")
    }
    fc_tool_enabled: bool = tool_config.get("flowcharts", {}).get("enabled", False)

    # --- Render -------------------------------------------------------------
    rich_result = _try_rich()
    if rich_result:
        from rich.panel import Panel

        console, Table, Text = rich_result

        # CLI tools table
        if cli_entries:
            cli_table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 1)
            )
            cli_table.add_column("Tool", style="bold cyan", no_wrap=True)
            cli_table.add_column("Description")
            for name, meta in cli_entries:
                desc = meta.description if meta else ""
                suffix = (
                    "  [bold yellow][confirm][/bold yellow]"
                    if registry.requires_confirmation(name)
                    else ""
                )
                cli_table.add_row(name, desc + suffix)
            console.print(
                Panel(
                    cli_table,
                    title=f"[bold]CLI tools[/bold] ({len(cli_entries)})",
                    border_style="blue",
                )
            )
        else:
            console.print("[dim]No CLI tools available.[/dim]")

        # Flowcharts section
        if all_fc_names or flowchart_entries:
            fc_table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 1)
            )
            fc_table.add_column("Flowchart", style="bold cyan", no_wrap=True)
            fc_table.add_column("Status")
            fc_table.add_column("Description")
            if not fc_tool_enabled:
                fc_table.add_column("[dim]note[/dim]")
            for fc_name in all_fc_names:
                status = (
                    "[green]active[/green]"
                    if fc_name in active_fc_names
                    else "[dim]inactive[/dim]"
                )
                meta = registry.get_tool(f"flowchart:{fc_name}")
                desc = (
                    meta.description
                    if meta
                    else f"Run the '{fc_name}' flowchart workflow"
                )
                if not fc_tool_enabled:
                    fc_table.add_row(
                        fc_name,
                        "[dim]disabled[/dim]",
                        desc,
                        "[dim]enable with: talos tools enable flowcharts[/dim]",
                    )
                else:
                    fc_table.add_row(fc_name, status, desc)
            title_note = (
                " [dim](flowchart tool disabled)[/dim]" if not fc_tool_enabled else ""
            )
            console.print(
                Panel(
                    fc_table,
                    title=f"[bold]Flowchart tools[/bold]{title_note}",
                    border_style="blue",
                )
            )
        else:
            console.print("[dim]No flowcharts configured.[/dim]")

        # Other virtual tools (web-research, memory, text2image)
        if virtual_entries:
            vt_table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 1)
            )
            vt_table.add_column("Tool", style="bold cyan", no_wrap=True)
            vt_table.add_column("Description")
            for name, meta in virtual_entries:
                desc = meta.description if meta else ""
                vt_table.add_row(name, desc)
            console.print(
                Panel(vt_table, title="[bold]Virtual tools[/bold]", border_style="blue")
            )

    else:
        # Plain-text fallback
        mode = tool_config.get("mode", "strict")
        print(f"Talos — Active Tools  (mode: {mode})\n")

        if cli_entries:
            print(f"CLI tools ({len(cli_entries)}):")
            for name, meta in cli_entries:
                desc = meta.description if meta else ""
                confirm_mark = (
                    "  [confirm]" if registry.requires_confirmation(name) else ""
                )
                print(f"  {name:<22} {desc}{confirm_mark}")
        else:
            print("  (no CLI tools available)")

        print()
        if all_fc_names:
            note = (
                "  (flowchart tool disabled — run: talos tools enable flowcharts)"
                if not fc_tool_enabled
                else ""
            )
            print(f"Flowchart tools:{note}")
            for fc_name in all_fc_names:
                active = "active" if fc_name in active_fc_names else "inactive"
                print(f"  {fc_name:<22} [{active}]")
        else:
            print("Flowchart tools:\n  (none configured)")

        if virtual_entries:
            print("\nVirtual tools:")
            for name, meta in virtual_entries:
                desc = meta.description if meta else ""
                print(f"  {name:<22} {desc}")


def cmd_list_all(config: TalosConfig, config_path: Path) -> None:
    """Print all known tools and their effective availability status.

    Combines:
    - All tools in the active tool_config.yaml lists (include, exclude,
      confirm, descriptions) plus any Talos-local allow/deny entries.
    - Discoverability check via ``shutil.which()`` for CLI tools.
    - Virtual tool (flowcharts, web-research, text2image) status.
    """
    tc = config.agent.tools
    cm = _make_config_manager(tc)

    # Load the merged effective config (includes talos allow/deny).
    _tmp = ToolRegistry(cm, providers=[])
    tool_config: dict[str, Any] = _tmp.config

    mode = tool_config.get("mode", "strict")
    include: list[str] = list(tool_config.get("include", []))
    exclude: list[str] = list(tool_config.get("exclude", []))
    confirm: list[str] = list(tool_config.get("confirm", []))
    descriptions: dict[str, str] = tool_config.get("descriptions", {})

    # Collect all CLI tool names (union of all lists + descriptions keys).
    all_cli: list[str] = []
    seen: set[str] = set()
    for name in include + exclude + confirm + list(descriptions.keys()):
        if name not in seen and name not in VIRTUAL_TOOLS:
            seen.add(name)
            all_cli.append(name)

    def _status(name: str) -> tuple[str, bool]:
        """Return (status_label, installed_on_path)."""
        installed = shutil.which(name) is not None
        in_exclude = name in exclude
        in_include = name in include
        in_confirm = name in confirm

        if in_exclude:
            return "blocked", installed
        if mode == "strict":
            if in_confirm:
                return "confirm", installed
            if in_include:
                if installed:
                    return "allowed", True
                return "not-found", False
            return "unlisted", installed
        else:
            # standard / all / permissive
            if in_confirm:
                return "confirm", installed
            return "allowed", installed

    # Status colours / markers for plain text.
    _label_display = {
        "allowed": ("✓ allowed", "[green]"),
        "confirm": ("? confirm", "[yellow]"),
        "blocked": ("✗ blocked", "[red]"),
        "not-found": ("! not-found", "[dim]"),
        "unlisted": ("~ unlisted", "[dim]"),
    }

    rich_result = _try_rich()
    if rich_result:
        console, Table, Text = rich_result
        table = Table(
            title=f"Talos — All Tools  [dim](mode: {mode})[/dim]",
            show_header=True,
        )
        table.add_column("Tool", style="bold", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Installed", no_wrap=True)
        table.add_column("Description")

        _style = {
            "allowed": "green",
            "confirm": "yellow",
            "blocked": "red",
            "not-found": "dim",
            "unlisted": "dim",
        }
        for name in sorted(all_cli):
            status, installed = _status(name)
            desc = descriptions.get(name, "")
            inst_str = "[green]yes[/green]" if installed else "[dim]no[/dim]"
            table.add_row(
                name,
                f"[{_style[status]}]{status}[/{_style[status]}]",
                inst_str,
                desc,
            )
        console.print(table)

        # Virtual tools section.
        vtable = Table(title="Virtual Tools", show_header=True)
        vtable.add_column("Tool", style="bold cyan", no_wrap=True)
        vtable.add_column("Status", no_wrap=True)
        _vmap = {
            "web-research": ("web_research", tool_config.get("web_research", {})),
            "text2image": ("text2image", tool_config.get("text2image", {})),
            "flowcharts": ("flowcharts", tool_config.get("flowcharts", {})),
        }
        for vname, (_, vcfg) in _vmap.items():
            enabled = vcfg.get("enabled", False)
            vstatus = "[green]enabled[/green]" if enabled else "[dim]disabled[/dim]"
            vtable.add_row(vname, vstatus)
        console.print(vtable)

    else:
        # Plain-text fallback.
        print(f"All tools  (mode: {mode})\n")
        print(f"  {'Tool':<24}  {'Status':<12}  {'Inst':<5}  Description")
        print("  " + "-" * 72)
        for name in sorted(all_cli):
            status, installed = _status(name)
            label, _ = _label_display[status]
            desc = descriptions.get(name, "")
            inst = "yes" if installed else "no"
            print(f"  {name:<24}  {label:<12}  {inst:<5}  {desc}")

        print("\nVirtual tools:")
        _vmap = {
            "web-research": tool_config.get("web_research", {}),
            "text2image": tool_config.get("text2image", {}),
            "flowcharts": tool_config.get("flowcharts", {}),
        }
        for vname, vcfg in _vmap.items():
            enabled = vcfg.get("enabled", False)
            vstatus = "enabled" if enabled else "disabled"
            print(f"  {vname:<24}  {vstatus}")


# ---------------------------------------------------------------------------
# Entry-point dispatcher
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, config_path: Path) -> int:
    """Dispatch the ``talos tools`` subcommand.

    Args:
        args: Parsed arguments (must have ``tools_action`` attribute).
        config_path: Path to the Talos config file.

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    # Load (or create) the config without running the wizard on missing file.
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(
            f"No Talos config found at {config_path}. "
            "Run 'talos config' first to create one.",
        )
        return 1

    action = args.tools_action

    if action in ("list", "ls"):
        cmd_list(config, config_path)
        return 0

    if action == "list-all":
        cmd_list_all(config, config_path)
        return 0

    if action == "enable":
        cmd_enable(args.tool_name, config, config_path)
        return 0

    if action == "disable":
        cmd_disable(args.tool_name, config, config_path)
        return 0

    # Unreachable under normal CLI use; argparse catches missing actions.
    print(f"Unknown tools action: {action!r}")
    return 1
