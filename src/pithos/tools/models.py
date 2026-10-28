"""Data models for the pithos tool calling system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    """Risk classification for a CLI command."""

    SAFE = "safe"
    REVIEW = "review"  # Requires user confirmation before executing
    BLOCK = "block"  # Denied outright; subprocess is never called


@dataclass
class SafetyVerdict:
    """Result of a CommandSafetyChecker evaluation."""

    level: RiskLevel
    reason: str  # Human-readable explanation shown to agent / logged


@dataclass
class ToolMetadata:
    """Metadata for a CLI tool or virtual tool (e.g. flowchart)."""

    name: str
    path: str
    description: str
    platform: str  # 'windows', 'unix', 'cross-platform'
    source: str  # 'system', 'manual', 'environment'
    tool_type: str = "cli"  # 'cli', 'flowchart'


@dataclass
class ToolCallRequest:
    """Represents a parsed tool call request from agent output."""

    command: str  # The full command to execute
    format: str  # Which format was matched (e.g. 'tool_call')
    raw_text: str  # Original matched text
    name: str = ""  # The tool name from the JSON call (e.g. 'run', 'memory')
    arguments: dict = field(default_factory=dict)  # Structured call arguments


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    command: str
    error_hint: Optional[str] = None  # Specific error guidance for agents
    safety_verdict: Optional["SafetyVerdict"] = None  # Set when safety checker ran
    image_paths: list = field(default_factory=list)  # Paths to generated image files
    report_paths: list = field(default_factory=list)  # Report files whose contents should be shown inline
    attachment_paths: list = field(
        default_factory=list
    )  # Files to surface as attachments only (contents NOT shown inline)
