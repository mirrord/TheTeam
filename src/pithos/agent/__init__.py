from .agent import Agent
from .ollama_agent import OllamaAgent

# EXLAgent and LlamacppAgent are real backends gated on optional packages
# (``exllamav2`` / ``llama-cpp-python``).  They are intentionally NOT
# re-exported here so users must opt in via an explicit submodule import
# (e.g. ``from pithos.agent.llamacpp_agent import LlamacppAgent``) and
# accept responsibility for installing the heavyweight backend deps.
# Importing the submodule itself always succeeds; instantiating the class
# raises :class:`ImportError` with installation guidance when the backend
# package is unavailable.
from .cli import interactive_chat, main
from .factory import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    canonical_provider,
    agent_class_for_provider,
    build_agent,
    build_agent_from_config,
)
from .history import ConversationStore, HistorySearchResult, MessageRecord
from .compaction import CompactionConfig, MemoryCompactor
from .recall import RecallConfig, AutoRecall
from .memory import MemoryModule
from ..context import Msg, UserMsg, AgentMsg, AgentContext

__all__ = [
    "Agent",
    "OllamaAgent",
    "DEFAULT_PROVIDER",
    "SUPPORTED_PROVIDERS",
    "canonical_provider",
    "agent_class_for_provider",
    "build_agent",
    "build_agent_from_config",
    "AgentContext",
    "Msg",
    "UserMsg",
    "AgentMsg",
    "ConversationStore",
    "HistorySearchResult",
    "MessageRecord",
    "CompactionConfig",
    "MemoryCompactor",
    "RecallConfig",
    "AutoRecall",
    "MemoryModule",
    "interactive_chat",
    "main",
]
