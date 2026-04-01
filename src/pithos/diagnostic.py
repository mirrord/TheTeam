"""Diagnostic checks for pithos environment, configuration, and connectivity."""

import argparse
import os
import sys
import platform
import time
from pathlib import Path

# OLLAMA_HOST may be a bind address like 0.0.0.0 which the Python client
# can't connect to.  Normalise to localhost *before* any ollama import so
# the default client gets the corrected value.

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if _host in ("0.0.0.0", "0.0.0.0:11434"):
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # Drop any cached ollama modules so they pick up the new env var
    for _k in [k for k in sys.modules if k == "ollama" or k.startswith("ollama.")]:
        del sys.modules[_k]


# ANSI color helpers (disabled when stdout is not a terminal)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOR else text


def _status(ok: bool) -> str:
    return _green("PASS") if ok else _red("FAIL")


def _warn_status() -> str:
    return _yellow("WARN")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> tuple[bool, str]:
    """Check that the Python version meets the minimum requirement (>=3.8)."""
    version = sys.version_info
    ok = version >= (3, 8)
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    if not ok:
        detail += " (requires >= 3.8)"
    return ok, detail


def check_ollama_package() -> tuple[bool, str]:
    """Check that the ollama Python package is importable."""
    try:
        import ollama  # noqa: F401

        return (
            True,
            f"ollama package found (version {getattr(ollama, '__version__', 'unknown')})",
        )
    except ImportError:
        return False, "ollama package not installed — run: pip install ollama"


def check_ollama_server() -> tuple[bool, str]:
    """Check connectivity to the Ollama server and list available models."""
    try:
        import ollama

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        t0 = time.monotonic()
        response = ollama.list()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        models = (
            [m.model for m in response.models] if hasattr(response, "models") else []
        )
        count = len(models)
        names = ", ".join(models[:5])
        if count > 5:
            names += f" … (+{count - 5} more)"
        detail = f"Connected to {host} in {elapsed_ms:.0f}ms — {count} model(s)"
        if names:
            detail += f": {names}"
        return True, detail
    except ImportError:
        return False, "ollama package not installed"
    except Exception as exc:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        return False, f"Cannot reach Ollama at {host} — {exc}"


def check_config_directory() -> tuple[bool, str]:
    """Check that the config directory exists and contains YAML files."""
    from .config_manager import ConfigManager, CONFIG_DIR_ENV_VAR

    env_val = os.environ.get(CONFIG_DIR_ENV_VAR)
    try:
        cm = ConfigManager()
    except Exception as exc:
        return False, f"Failed to initialize ConfigManager: {exc}"

    config_dir = cm.config_dir
    if not config_dir.exists():
        return False, f"Config directory does not exist: {config_dir}"

    namespaces = list(cm.get_registered_namespaces())
    total_configs = sum(len(v) for v in cm.configs.values())

    detail = (
        f"{config_dir} — {total_configs} config(s) in {len(namespaces)} namespace(s)"
    )
    if namespaces:
        detail += f" [{', '.join(namespaces)}]"
    if env_val:
        detail += f" (via {CONFIG_DIR_ENV_VAR})"
    return True, detail


def check_agent_configs() -> tuple[bool, str]:
    """Check that at least one agent configuration is registered."""
    from .config_manager import ConfigManager

    try:
        cm = ConfigManager()
    except Exception as exc:
        return False, f"ConfigManager error: {exc}"

    agents = list(cm.get_registered_agent_names())
    if not agents:
        return False, "No agent configs found in configs/agents/"
    return True, f"{len(agents)} agent config(s): {', '.join(agents)}"


def check_flowchart_configs() -> tuple[bool, str]:
    """Check that flowchart configurations are registered."""
    from .config_manager import ConfigManager

    try:
        cm = ConfigManager()
    except Exception as exc:
        return False, f"ConfigManager error: {exc}"

    flowcharts = list(cm.get_registered_flowchart_names())
    if not flowcharts:
        return False, "No flowchart configs found in configs/flowcharts/"
    return (
        True,
        f"{len(flowcharts)} flowchart config(s): {', '.join(flowcharts[:8])}"
        + (f" … (+{len(flowcharts) - 8} more)" if len(flowcharts) > 8 else ""),
    )


def check_chromadb() -> tuple[bool, str]:
    """Check ChromaDB availability for vector memory."""
    try:
        import chromadb  # noqa: F401

        version = getattr(chromadb, "__version__", "unknown")
        return True, f"chromadb {version} available"
    except ImportError:
        return False, "chromadb not installed — vector memory will be unavailable"


def check_yaml() -> tuple[bool, str]:
    """Check that PyYAML is installed."""
    try:
        import yaml  # noqa: F401

        version = getattr(yaml, "__version__", "unknown")
        return True, f"PyYAML {version}"
    except ImportError:
        return False, "PyYAML not installed — run: pip install pyyaml"


def check_networkx() -> tuple[bool, str]:
    """Check that NetworkX is installed (required for flowcharts)."""
    try:
        import networkx  # noqa: F401

        return True, f"networkx {networkx.__version__}"
    except ImportError:
        return False, "networkx not installed — run: pip install networkx"


def check_flask() -> tuple[bool, str]:
    """Check that Flask and SocketIO are installed (required for web interface)."""
    missing = []
    versions = []
    for pkg_name, import_name in [
        ("flask", "flask"),
        ("flask-socketio", "flask_socketio"),
        ("flask-cors", "flask_cors"),
        ("eventlet", "eventlet"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            versions.append(f"{pkg_name} {ver}")
        except ImportError:
            missing.append(pkg_name)

    if missing:
        return False, f"Missing web packages: {', '.join(missing)}"
    return True, "; ".join(versions)


def check_tool_registry() -> tuple[bool, str]:
    """Check that ToolRegistry can initialise and discover tools."""
    from .config_manager import ConfigManager
    from .tools import ToolRegistry

    try:
        cm = ConfigManager()
        registry = ToolRegistry(cm)
        count = len(registry.tools)
        names = ", ".join(list(registry.tools.keys())[:8])
        if count > 8:
            names += f" … (+{count - 8} more)"
        return True, (
            f"{count} tool(s) discovered: {names}"
            if names
            else f"{count} tool(s) discovered"
        )
    except Exception as exc:
        return False, f"ToolRegistry init failed: {exc}"


def check_conversation_store() -> tuple[bool, str]:
    """Check that the conversation history database can be opened."""
    from .agent.history import ConversationStore

    try:
        store = ConversationStore()
        db_path = store._db_path
        sessions = store.list_sessions()
        count = len(sessions)
        store.close()
        return True, f"Database at {db_path} — {count} session(s)"
    except Exception as exc:
        return False, f"ConversationStore error: {exc}"


def check_data_directory() -> tuple[bool, str]:
    """Check that the data directory exists and is writable."""
    data_dir = Path.cwd() / "data"
    if not data_dir.exists():
        return False, f"Data directory does not exist: {data_dir}"
    if not os.access(data_dir, os.W_OK):
        return False, f"Data directory not writable: {data_dir}"
    return True, str(data_dir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# All checks in execution order — (name, function, is_critical)
ALL_CHECKS: list[tuple[str, callable, bool]] = [
    ("Python version", check_python_version, True),
    ("PyYAML", check_yaml, True),
    ("NetworkX", check_networkx, True),
    ("Ollama package", check_ollama_package, True),
    ("Ollama server", check_ollama_server, True),
    ("ChromaDB", check_chromadb, False),
    ("Flask / web stack", check_flask, False),
    ("Config directory", check_config_directory, True),
    ("Agent configs", check_agent_configs, False),
    ("Flowchart configs", check_flowchart_configs, False),
    ("Tool registry", check_tool_registry, False),
    ("Conversation store", check_conversation_store, False),
    ("Data directory", check_data_directory, False),
]


def run_diagnostics(verbose: bool = False) -> bool:
    """Run all diagnostic checks and print results.

    Returns True if all critical checks pass.
    """
    print(_bold("pithos diagnostic"))
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {sys.executable}")
    print(f"CWD:      {Path.cwd()}")
    print()

    passed = 0
    failed = 0
    warned = 0
    critical_fail = False

    for name, check_fn, critical in ALL_CHECKS:
        try:
            ok, detail = check_fn()
        except Exception as exc:
            ok, detail = False, f"Unexpected error: {exc}"

        if ok:
            passed += 1
            marker = _status(True)
        elif not critical:
            warned += 1
            marker = _warn_status()
        else:
            failed += 1
            critical_fail = True
            marker = _status(False)

        label = f"{'*' if critical else ' '} {name}"
        print(f"  [{marker}] {label}")
        if verbose or not ok:
            print(f"         {detail}")

    print()
    summary_parts = [_green(f"{passed} passed")]
    if warned:
        summary_parts.append(_yellow(f"{warned} warning(s)"))
    if failed:
        summary_parts.append(_red(f"{failed} failed"))
    print(f"  {', '.join(summary_parts)}  (* = critical)")

    if critical_fail:
        print(
            f"\n{_red('Some critical checks failed.')} Resolve the issues above before proceeding."
        )
    else:
        print(f"\n{_green('All critical checks passed.')}")

    return not critical_fail


def main() -> None:
    """CLI entry point for `pithos-diagnostic`."""
    parser = argparse.ArgumentParser(
        prog="pithos-diagnostic",
        description="Run diagnostic checks on the pithos environment.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show details for passing checks too.",
    )
    args = parser.parse_args()
    ok = run_diagnostics(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
