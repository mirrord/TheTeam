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
from .history import ConversationStore, HistorySearchResult, MessageRecord
from .compaction import CompactionConfig, MemoryCompactor
from .recall import RecallConfig, AutoRecall
from ..context import Msg, UserMsg, AgentMsg, AgentContext

__all__ = [
    "Agent",
    "OllamaAgent",
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
    "interactive_chat",
    "main",
]
