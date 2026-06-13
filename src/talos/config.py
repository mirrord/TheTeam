"""Talos configuration management.

Provides dataclasses, YAML load/save, an interactive first-run wizard,
and an :func:`build_agent` helper that wires a :class:`TalosConfig` into
a fully-configured :class:`pithos.OllamaAgent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import yaml

from pithos import OllamaAgent, ConfigManager

DEFAULT_CONFIG_PATH = Path.home() / ".talos" / "config.yaml"

# ---- Default configuration values ----------------------------------------
# Defined once here so that dataclass field defaults and wizard prompts always
# stay in sync.  Wizard-specific "recommended" values that intentionally differ
# from the conservative programmatic defaults are prefixed with _WIZARD_.

DEFAULT_MODEL: str = "glm-4.7-flash"
DEFAULT_SYSTEM_PROMPT: str = (
    "You are Talos, a helpful AI assistant. Answer concisely in 20 words or less."
)
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOOLS: bool = False
DEFAULT_TOOLS_MODE: str = "include"  # "all" | "include" | "exclude" | "confirm"
DEFAULT_TOOLS_AUTO_LOOP: bool = False
DEFAULT_TOOLS_MAX_ITERATIONS: int = 5
DEFAULT_MEMORY: bool = False
DEFAULT_MEMORY_COMPACTION: bool = False
DEFAULT_MEMORY_COMPACTION_THRESHOLD: int = 20
DEFAULT_MEMORY_RECALL: bool = False
DEFAULT_MEMORY_HISTORY: bool = False
DEFAULT_DEVICE: str = "cuda"  # "cuda" or "cpu"
DEFAULT_WAKE_WORD: str = "hey talos"
DEFAULT_WAKE_WORD_CHUNK_SECONDS: float = (
    2.0  # audio window whisper scans for the wake word
)
DEFAULT_TTS_VOICE_PRESET: str = "af_heart"
DEFAULT_SILENCE_THRESHOLD: float = 500.0
DEFAULT_SILENCE_DURATION: float = 1.5
DEFAULT_MAX_RECORD_SECONDS: float = 30.0

# Recommended values the wizard suggests — richer setup than the conservative
# programmatic defaults above.
_WIZARD_TOOLS: bool = True
_WIZARD_MEMORY: bool = True
_WIZARD_DEVICE: str = "cuda"
# ---------------------------------------------------------------------------


@dataclass
class ToolsConfig:
    """Tool-calling settings for the Talos agent.

    Attributes:
        enabled: Master switch — when False, no tools are loaded.
        mode: Tool discovery filter — ``"all"``, ``"include"``,
            ``"exclude"``, or ``"confirm"``.  Overrides the ``mode`` field in
            ``configs/tools/tool_config.yaml`` so users can grant Talos
            access to every discovered tool without editing the shared
            tool config.
        auto_loop: Automatically continue the conversation after a tool
            call so the agent can react to the result without a new user
            message.
        max_iterations: Safety cap on auto-loop iterations per turn.
    """

    enabled: bool = DEFAULT_TOOLS
    mode: str = DEFAULT_TOOLS_MODE
    auto_loop: bool = DEFAULT_TOOLS_AUTO_LOOP
    max_iterations: int = DEFAULT_TOOLS_MAX_ITERATIONS
    web_research: Optional[dict[str, bool]] = None  # enable web search tool
    flowcharts: Optional[dict[str, Any]] = None


@dataclass
class MemoryConfig:
    """Memory and conversation-augmentation settings for the Talos agent.

    Attributes:
        enabled: Master switch for the vector memory store.  Required for
            ``recall`` to do anything useful.
        persist_directory: Optional override for the ChromaDB / history
            storage location.  Defaults are managed by pithos.
        compaction: Enable automatic context compaction (summarises older
            messages once the threshold is reached).
        compaction_threshold: Message count that triggers compaction.
        recall: Enable automatic RAG injection of relevant memories /
            history before each user turn.
        history: Enable persistent conversation history (SQLite + vector
            index).  Required as a recall source for past conversations.
        tag_suggestions_model: Optional Ollama model name; when set,
            memory entries are tagged automatically by an LLM at store
            time.  ``None`` disables tag suggestions.
    """

    enabled: bool = DEFAULT_MEMORY
    persist_directory: Optional[str] = None
    compaction: bool = DEFAULT_MEMORY_COMPACTION
    compaction_threshold: int = DEFAULT_MEMORY_COMPACTION_THRESHOLD
    recall: bool = DEFAULT_MEMORY_RECALL
    history: bool = DEFAULT_MEMORY_HISTORY
    tag_suggestions_model: Optional[str] = None


@dataclass
class AgentConfig:
    """Agent settings used to build the underlying pithos agent."""

    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = DEFAULT_TEMPERATURE
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Build an :class:`AgentConfig` from a plain dict.

        Nested ``tools`` and ``memory`` entries are accepted as dicts and
        converted to their respective dataclasses.
        """
        data = dict(data or {})
        tools_data = data.pop("tools", None) or {}
        memory_data = data.pop("memory", None) or {}
        return cls(
            **data,
            tools=ToolsConfig(**tools_data),
            memory=MemoryConfig(**memory_data),
        )


@dataclass
class VoiceConfig:
    """Voice interface settings."""

    device: str = DEFAULT_DEVICE  # "cuda" or "cpu"
    microphone_device: Optional[int] = None  # None = system default
    wake_word: str = DEFAULT_WAKE_WORD  # phrase whisper must detect to activate
    wake_word_chunk_seconds: float = DEFAULT_WAKE_WORD_CHUNK_SECONDS
    tts_voice_preset: str = DEFAULT_TTS_VOICE_PRESET
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD  # RMS energy threshold
    silence_duration: float = (
        DEFAULT_SILENCE_DURATION  # seconds of silence to end recording
    )
    max_record_seconds: float = DEFAULT_MAX_RECORD_SECONDS


@dataclass
class TelegramConfig:
    """Telegram interface settings."""

    bot_token: str = ""


@dataclass
class TalosConfig:
    """Top-level Talos configuration."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": asdict(self.agent),
            "voice": asdict(self.voice),
            "telegram": asdict(self.telegram),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TalosConfig":
        data = data or {}
        return cls(
            agent=AgentConfig.from_dict(data.get("agent") or {}),
            voice=VoiceConfig(**(data.get("voice") or {})),
            telegram=TelegramConfig(**(data.get("telegram") or {})),
        )


def load_config(path: Optional[Path] = None) -> TalosConfig:
    """Load a Talos config from YAML, applying defaults for missing keys."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return TalosConfig.from_dict(data)


def save_config(config: TalosConfig, path: Optional[Path] = None) -> Path:
    """Serialize a Talos config to YAML, creating parent dirs as needed."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    return path


def run_wizard() -> TalosConfig:
    """Interactive first-run wizard to create a Talos config.

    Uses ``rich.prompt`` when available; falls back to plain ``input()``.
    """
    try:
        from rich.console import Console
        from rich.prompt import Prompt, Confirm, FloatPrompt
    except ImportError:  # pragma: no cover - fallback path
        return _wizard_fallback()

    console = Console()
    console.print("\n[bold cyan]Talos first-run setup[/bold cyan]\n")

    # Agent
    console.print("[bold]Agent[/bold]")
    model = Prompt.ask("Model name (Ollama)", default=DEFAULT_MODEL)
    system_prompt = Prompt.ask(
        "System prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    temperature = FloatPrompt.ask("Temperature", default=DEFAULT_TEMPERATURE)
    tools = Confirm.ask("Enable CLI tools?", default=_WIZARD_TOOLS)
    memory = Confirm.ask("Enable vector memory (ChromaDB)?", default=_WIZARD_MEMORY)

    # Voice
    console.print("\n[bold]Voice interface[/bold]")
    device = Prompt.ask(
        "Compute device", choices=["cpu", "cuda"], default=_WIZARD_DEVICE
    )
    wake_word = Prompt.ask(
        "Wake word (phrase to say to activate the assistant)",
        default=DEFAULT_WAKE_WORD,
    )

    # Telegram
    console.print("\n[bold]Telegram interface[/bold] (leave blank to skip)")
    bot_token = Prompt.ask("Telegram bot token", default="", show_default=False)

    return TalosConfig(
        agent=AgentConfig(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            tools=ToolsConfig(enabled=tools),
            memory=MemoryConfig(enabled=memory),
        ),
        voice=VoiceConfig(
            device=device,
            wake_word=wake_word,
        ),
        telegram=TelegramConfig(bot_token=bot_token),
    )


def _wizard_fallback() -> TalosConfig:
    """Minimal-dependency wizard if ``rich`` is not installed."""
    print("\nTalos first-run setup\n")

    def ask(prompt: str, default: str) -> str:
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default

    def ask_bool(prompt: str, default: bool) -> bool:
        d = "y" if default else "n"
        val = input(f"{prompt} (y/n) [{d}]: ").strip().lower() or d
        return val.startswith("y")

    model = ask("Model name (Ollama)", DEFAULT_MODEL)
    system_prompt = ask("System prompt", DEFAULT_SYSTEM_PROMPT)
    temperature = float(ask("Temperature", str(DEFAULT_TEMPERATURE)))
    tools = ask_bool("Enable CLI tools?", _WIZARD_TOOLS)
    memory = ask_bool("Enable vector memory (ChromaDB)?", _WIZARD_MEMORY)
    device = ask("Compute device (cpu/cuda)", _WIZARD_DEVICE)
    wake_word = ask("Wake word (phrase to activate the assistant)", DEFAULT_WAKE_WORD)
    bot_token = ask("Telegram bot token (blank to skip)", "")

    return TalosConfig(
        agent=AgentConfig(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            tools=ToolsConfig(enabled=tools),
            memory=MemoryConfig(enabled=memory),
        ),
        voice=VoiceConfig(
            device=device,
            wake_word=wake_word,
        ),
        telegram=TelegramConfig(bot_token=bot_token),
    )


def ensure_config(
    path: Optional[Path] = None, force_wizard: bool = False
) -> tuple[TalosConfig, Path]:
    """Load the Talos config, running the first-run wizard if absent.

    Args:
        path: Optional path override (defaults to ``~/.talos/config.yaml``).
        force_wizard: If True, always run the wizard regardless of existence.

    Returns:
        ``(config, path)`` tuple. The config file is created on disk when the
        wizard runs.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if force_wizard or not path.exists():
        config = run_wizard()
        save_config(config, path)
        print(f"Saved Talos config to {path}")
        return config, path
    return load_config(path), path


def build_agent(config: TalosConfig) -> OllamaAgent:
    """Construct a configured :class:`OllamaAgent` from a Talos config.

    Wires up tools (with optional discovery-mode override), the vector
    memory store, automatic compaction / recall / persistent history, and
    LLM-driven memory tag suggestions, according to the user's Talos
    config.  Each feature is independently toggleable.
    """
    agent = OllamaAgent(
        default_model=config.agent.model,
        agent_name="talos",
        system_prompt=config.agent.system_prompt,
        temperature=config.agent.temperature,
    )

    tc = config.agent.tools
    mc = config.agent.memory

    if not (tc.enabled or mc.enabled):
        return agent

    # ConfigManager is required for tools/memory.  When the user picked a
    # tool mode other than the tool_config.yaml default ("include"), wrap
    # the manager to inject the override without mutating the YAML file.
    # Collect Talos-level overrides that must win over tool_config.yaml.
    tool_overrides: dict[str, Any] = {}
    if tc.flowcharts is not None:
        tool_overrides["flowcharts"] = tc.flowcharts
    if tc.web_research is not None:
        tool_overrides["web_research"] = tc.web_research

    if tc.enabled and (tc.mode != DEFAULT_TOOLS_MODE or tool_overrides):
        cm_kwargs: dict[str, Any] = {"tool_mode_override": tc.mode}
        if tool_overrides:
            cm_kwargs["tool_config_overrides"] = tool_overrides
        cm: ConfigManager = _ModeOverrideConfigManager(**cm_kwargs)
    else:
        cm = ConfigManager()

    if tc.enabled:
        agent.enable_tools(
            cm,
            auto_loop=tc.auto_loop,
            max_iterations=tc.max_iterations,
        )

    if mc.enabled:
        agent.enable_memory(cm, persist_directory=mc.persist_directory)
        if mc.compaction:
            from pithos.agent.compaction import CompactionConfig

            agent.enable_compaction(CompactionConfig(threshold=mc.compaction_threshold))
        if mc.recall:
            from pithos.agent.recall import RecallConfig

            agent.enable_recall(RecallConfig())
        if mc.history:
            agent.enable_history(persist_directory=mc.persist_directory)
        if mc.tag_suggestions_model:
            agent.enable_tag_suggestions(model=mc.tag_suggestions_model)

    return agent


class _ModeOverrideConfigManager(ConfigManager):
    """ConfigManager that overrides tool discovery settings at read time.

    Used by :func:`build_agent` to honour ``ToolsConfig`` values without
    modifying the shared ``configs/tools/tool_config.yaml`` file.  The
    ``tool_mode_override`` always wins over the YAML ``mode`` field;
    ``tool_config_overrides`` (if provided) are shallow-merged on top of
    the full config so that Talos-level feature flags (e.g.
    ``flowcharts.enabled``, ``web_research.enabled``) always take
    precedence over registry defaults.
    """

    def __init__(
        self,
        tool_mode_override: str,
        config_dir: Optional[str] = None,
        tool_config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config_dir)
        self._tool_mode_override = tool_mode_override
        self._tool_config_overrides: dict[str, Any] = tool_config_overrides or {}

    def get_config(
        self, config_name: str, namespace: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        cfg = super().get_config(config_name, namespace)
        if cfg is not None and config_name == "tool_config" and namespace == "tools":
            cfg = dict(cfg)
            cfg["mode"] = self._tool_mode_override
            cfg.update(self._tool_config_overrides)
        return cfg


__all__ = [
    "AgentConfig",
    "ToolsConfig",
    "MemoryConfig",
    "VoiceConfig",
    "TelegramConfig",
    "TalosConfig",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOOLS",
    "DEFAULT_TOOLS_MODE",
    "DEFAULT_TOOLS_AUTO_LOOP",
    "DEFAULT_TOOLS_MAX_ITERATIONS",
    "DEFAULT_MEMORY",
    "DEFAULT_MEMORY_COMPACTION",
    "DEFAULT_MEMORY_COMPACTION_THRESHOLD",
    "DEFAULT_MEMORY_RECALL",
    "DEFAULT_MEMORY_HISTORY",
    "DEFAULT_DEVICE",
    "DEFAULT_WAKE_WORD",
    "DEFAULT_WAKE_WORD_CHUNK_SECONDS",
    "DEFAULT_TTS_VOICE_PRESET",
    "DEFAULT_SILENCE_THRESHOLD",
    "DEFAULT_SILENCE_DURATION",
    "DEFAULT_MAX_RECORD_SECONDS",
    "load_config",
    "save_config",
    "run_wizard",
    "ensure_config",
    "build_agent",
]
