"""Demo: Expanded Tool Calling — Terminal Programs & Flowcharts as Tools.

This script demonstrates the two new tool-calling capabilities:

1. **Terminal / Shell tools** — agents can now invoke shell interpreters
   (powershell, bash, cmd …) and dozens of common CLI utilities
   (ping, curl, git, tree, diff, etc.) directly.

2. **Flowchart tools** — agents can invoke any registered pithos flowchart
   as a tool via ``RUN: flowchart <name> <input>``, receiving the
   flowchart's output as a tool result.

Run:
    python demos/tools_demo.py

Requirements:
    - Ollama running locally with a model available (default: glm-4.7-flash)
"""

import sys
import textwrap
from pathlib import Path

# Ensure the src directory is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pithos import OllamaAgent, ConfigManager
from pithos.tools import (
    ToolRegistry,
    ToolExecutor,
    ToolCallExtractor,
    FlowchartToolExecutor,
    format_tool_result_for_agent,
)

# ── helpers ──────────────────────────────────────────────────────────────────

DIVIDER = "-" * 70
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{DIVIDER}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{DIVIDER}{RESET}")


def step(label: str) -> None:
    print(f"\n{BOLD}{GREEN}> {label}{RESET}")


def info(text: str) -> None:
    for line in textwrap.wrap(text, width=68):
        print(f"  {DIM}{line}{RESET}")


def show_result(result) -> None:
    status = f"{GREEN}OK{RESET}" if result.success else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}Command:{RESET} {result.command}")
    print(
        f"  {BOLD}Status:{RESET}  {status}  (exit {result.exit_code}, "
        f"{result.execution_time:.2f}s)"
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines()[:15]:
            print(f"    {line}")
        if len(result.stdout.strip().splitlines()) > 15:
            print(
                f"    {DIM}... ({len(result.stdout.strip().splitlines()) - 15} more lines){RESET}"
            )
    if result.stderr:
        print(f"  {RED}stderr:{RESET} {result.stderr.strip()[:200]}")
    if result.error_hint:
        print(f"  {YELLOW}hint:{RESET} {result.error_hint}")


def ask(prompt: str, default: str) -> str:
    answer = input(f"  {prompt} [{default}]: ").strip()
    return answer if answer else default


# ── Part 1: Tool Registry Discovery ─────────────────────────────────────────


def demo_tool_discovery() -> ConfigManager:
    header("Part 1 — Tool Registry Discovery")
    info(
        "The ToolRegistry scans your system PATH and merges the "
        "tool_config.yaml include-list to build a catalogue of every "
        "tool an agent is allowed to use. Shell interpreters and common "
        "terminal utilities are now included by default."
    )

    cm = ConfigManager()
    ToolRegistry.invalidate_cache()
    registry = ToolRegistry(cm)

    step("All registered tools")
    cli_count = sum(1 for t in registry.tools.values() if t.tool_type == "cli")
    fc_count = sum(1 for t in registry.tools.values() if t.tool_type == "flowchart")
    print(
        f"  {BOLD}{cli_count}{RESET} CLI tools, "
        f"{BOLD}{fc_count}{RESET} flowchart tools\n"
    )

    step("CLI tools actually found on this system")
    for name in sorted(registry.tools):
        t = registry.tools[name]
        if t.tool_type == "cli":
            print(f"  {BOLD}{name:16s}{RESET} {DIM}{t.description[:60]}{RESET}")

    step("Flowchart tools")
    for name in sorted(registry.tools):
        t = registry.tools[name]
        if t.tool_type == "flowchart":
            print(f"  {BOLD}{name:35s}{RESET} {DIM}{t.description[:50]}{RESET}")

    step("Formatted tool list (as injected into agent system prompt)")
    print()
    print(registry.get_tool_list_text())

    return cm


# ── Part 2: Direct Tool Execution ───────────────────────────────────────────


def demo_direct_execution(cm: ConfigManager) -> None:
    header("Part 2 — Direct CLI Tool Execution")
    info(
        "Each tool call is validated against the registry, run in a "
        "subprocess with timeout protection, and its output captured."
    )

    registry = ToolRegistry(cm)
    executor = ToolExecutor(timeout=15, max_output_size=5000)

    commands = [
        "python --version",
        "git --version",
        "hostname",
    ]

    for cmd in commands:
        step(f"Running: {cmd}")
        result = executor.run(cmd, registry)
        show_result(result)

    # Show a blocked command
    step("Attempting a blocked command: rm -rf /")
    result = executor.run("rm -rf /", registry)
    show_result(result)


# ── Part 3: Extractor Patterns ──────────────────────────────────────────────


def demo_extraction() -> None:
    header("Part 3 — Tool Call Extraction from Agent Text")
    info(
        "The ToolCallExtractor recognises multiple formats so that agents "
        "can express tool calls however is natural for them."
    )

    extractor = ToolCallExtractor()

    sample_text = """\
Let me check your Python version first.
RUN: python --version

I'll also look at the git log.
tool(git log --oneline -5)

And let me verify the directory contents:
[RUN]echo hello world[/RUN]
"""

    step("Parsing this agent output")
    print()
    for line in sample_text.strip().splitlines():
        print(f"  {DIM}|{RESET} {line}")

    calls = extractor.extract(sample_text)
    print()
    step(f"Extracted {len(calls)} tool call(s)")
    for i, call in enumerate(calls, 1):
        print(f"  {BOLD}{i}.{RESET} format={call.format:8s}  command={call.command}")


# ── Part 4: Flowchart Tool ──────────────────────────────────────────────────


def demo_flowchart_tool(cm: ConfigManager) -> None:
    header("Part 4 — Flowcharts as Tools")
    info(
        "Agents can now invoke any registered flowchart via "
        '"RUN: flowchart <name> <input>". The FlowchartToolExecutor '
        "discovers all registered flowcharts and wraps their execution "
        "into the standard ToolResult interface."
    )

    fc_exec = FlowchartToolExecutor(cm, timeout=120, max_steps=100)

    step("Available flowcharts")
    for name in fc_exec.list_flowcharts():
        print(f"  {BOLD}{name}{RESET}")

    step("Flowchart ToolMetadata entries")
    for key, meta in fc_exec.discover_flowcharts().items():
        print(f"  {BOLD}{key:35s}{RESET} type={meta.tool_type}  source={meta.source}")


# ── Part 5: Agent with Full Tool Suite ──────────────────────────────────────


def demo_agent_with_tools(cm: ConfigManager) -> None:
    header("Part 5 — Agent with Tools Enabled (Interactive)")
    info(
        "This creates an OllamaAgent with tools enabled (including "
        "flowchart tools) and starts an interactive chat. The agent can "
        "use RUN:/EXEC:/tool() to invoke any allowed tool. Try asking "
        "it to run shell commands or flowcharts!"
    )

    model = ask("Ollama model to use", "glm-4.7-flash")

    agent = OllamaAgent(
        default_model=model,
        agent_name="tools-demo",
        system_prompt=(
            "You are a helpful assistant with access to CLI tools and flowcharts. "
            "Use tools when they would help answer the user's question. "
            "Always explain what you're doing before and after using a tool."
        ),
    )
    agent.enable_tools(cm, auto_loop=True, max_iterations=3)

    fc_exec = agent.flowchart_executor
    if fc_exec:
        info(
            f"Flowchart tools enabled — {len(fc_exec.list_flowcharts())} flowcharts available."
        )
    else:
        info("Flowchart tools not enabled (check tool_config.yaml).")

    print()
    step("System prompt (first 500 chars)")
    ctx = agent.contexts["default"]
    sp = ctx.get_system_prompt()[:500]
    for line in sp.splitlines():
        print(f"  {DIM}{line}{RESET}")
    if len(ctx.get_system_prompt()) > 500:
        print(f"  {DIM}... (truncated){RESET}")

    print(f"\n  {BOLD}Type 'quit' to exit.{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        try:
            response = agent.send(user_input)
            print(f"\n{BOLD}{CYAN}Agent:{RESET} {response}\n")
        except Exception as exc:
            print(f"\n{RED}Error:{RESET} {exc}\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  pithos — Expanded Tools Demo{RESET}")
    print(f"{BOLD}{CYAN}  Terminal programs & Flowcharts as agent tools{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")

    # Parts 1–4 are non-interactive and require no LLM
    cm = demo_tool_discovery()
    demo_direct_execution(cm)
    demo_extraction()
    demo_flowchart_tool(cm)

    # Part 5 is interactive and requires Ollama
    print()
    go = ask("Launch interactive agent chat? (y/n)", "y")
    if go.lower().startswith("y"):
        demo_agent_with_tools(cm)

    print(f"\n{BOLD}{CYAN}Done!{RESET}")


if __name__ == "__main__":
    main()
