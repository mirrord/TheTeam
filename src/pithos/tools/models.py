"""Data models for the pithos tool calling system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolMetadata:
    """Metadata for a CLI tool."""

    name: str
    path: str
    description: str
    platform: str  # 'windows', 'unix', 'cross-platform'
    source: str  # 'system', 'manual', 'environment'


@dataclass
class ToolCallRequest:
    """Represents a parsed tool call request from agent output."""

    command: str  # The full command to execute
    format: str  # Which format was matched: 'cli', 'function', 'bracket', 'legacy'
    raw_text: str  # Original matched text


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
