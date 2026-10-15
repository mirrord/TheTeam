"""pithos - Agentic LLM interaction framework."""

from .agent import Agent, OllamaAgent, build_agent, build_agent_from_config

# EXLAgent / LlamacppAgent are optional backends gated on heavyweight deps.
# Select a backend per-model via the ``provider`` config key and the
# ``build_agent`` / ``build_agent_from_config`` factory helpers, or import the
# classes explicitly from pithos.agent.exl_agent / pithos.agent.llamacpp_agent.
from .agent.history import ConversationStore, HistorySearchResult, MessageRecord
from .agent.compaction import CompactionConfig, MemoryCompactor
from .agent.recall import RecallConfig, AutoRecall
from .context import AgentContext, Msg, UserMsg, AgentMsg
from .tools import MemoryOpRequest, MemoryOpExtractor
from .team import AgentTeam, TeamContext
from .flowchart import Flowchart, ProgressEvent, EdgeInfo
from .flownode import (
    FlowNode,
    PromptNode,
    CustomNode,
    InputNode,
    OutputNode,
    ChatInputNode,
    ChatOutputNode,
    FileInputNode,
    FileOutputNode,
)
from .coding_nodes import (
    RouterNode,
    JsonParseNode,
    ListFilesNode,
    EditFileNode,
    GitNode,
)
from .conditions import (
    Condition,
    CountCondition,
    RegexCondition,
    AlwaysCondition,
    PredicateCondition,
)
from .config_manager import ConfigManager
from .message import Message, MessageRouter, NodeInputState
from .metrics import MetricsCollector

__all__ = [
    "Agent",
    "OllamaAgent",
    "build_agent",
    "build_agent_from_config",
    "AgentContext",
    "Msg",
    "UserMsg",
    "AgentMsg",
    "MemoryOpRequest",
    "MemoryOpExtractor",
    "AgentTeam",
    "TeamContext",
    "Flowchart",
    "FlowNode",
    "PromptNode",
    "CustomNode",
    "InputNode",
    "OutputNode",
    "ChatInputNode",
    "ChatOutputNode",
    "FileInputNode",
    "FileOutputNode",
    "RouterNode",
    "JsonParseNode",
    "ListFilesNode",
    "EditFileNode",
    "GitNode",
    "Condition",
    "CountCondition",
    "RegexCondition",
    "AlwaysCondition",
    "PredicateCondition",
    "ConfigManager",
    "Message",
    "MessageRouter",
    "NodeInputState",
    "ProgressEvent",
    "EdgeInfo",
    "ConversationStore",
    "HistorySearchResult",
    "MessageRecord",
    "CompactionConfig",
    "MemoryCompactor",
    "RecallConfig",
    "AutoRecall",
    "MetricsCollector",
]
