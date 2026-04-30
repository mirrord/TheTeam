from .agent import Agent
from .ollama_agent import OllamaAgent

# EXLAgent and LlamacppAgent are stub backends that raise NotImplementedError on
# construction. They are intentionally not re-exported here so that callers must
# opt in via an explicit submodule import (and acknowledge the stub status).
# See docs/ARCHITECTURE.md → Roadmap.
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
