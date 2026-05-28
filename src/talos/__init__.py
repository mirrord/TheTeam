"""Talos — local-first AI assistant interfaces (shell / voice / telegram).

Talos is built on the pithos agent framework. Configuration is loaded
from ``~/.talos/config.yaml`` (created via interactive wizard on first
run). Launch via the ``talos`` CLI:

    talos shell        # interactive stdin/stdout chat
    talos voice        # wake-word + speech-to-speech
    talos telegram     # telegram bot
"""

from .config import (
    AgentConfig,
    ToolsConfig,
    MemoryConfig,
    VoiceConfig,
    TelegramConfig,
    TalosConfig,
    DEFAULT_CONFIG_PATH,
    load_config,
    save_config,
    ensure_config,
    build_agent,
)
from .utils import clean_agent_response

__all__ = [
    "AgentConfig",
    "ToolsConfig",
    "MemoryConfig",
    "VoiceConfig",
    "TelegramConfig",
    "TalosConfig",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "save_config",
    "ensure_config",
    "build_agent",
    "clean_agent_response",
]
