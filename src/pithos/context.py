"""Agent context and message classes for managing conversation state."""

from dataclasses import dataclass
from typing import Optional, Any
from copy import deepcopy

from .config_manager import ConfigManager


@dataclass
class Msg:
    """Message container for agent conversations."""

    role: str
    content: str

    def __setitem__(self, key: str, value: Any) -> None:
        """Set message attribute by key."""
        self.__dict__[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get message attribute by key."""
        return self.__dict__[key]

    def raw(self) -> dict[str, str]:
        """Return message as raw dictionary."""
        return self.__dict__


class UserMsg(Msg):
    """User message."""

    def __init__(self, content: str):
        """Initialize user message.

        Args:
            content: Message content.
        """
        super().__init__("user", content)


class AgentMsg(Msg):
    """Assistant/agent message."""

    def __init__(self, content: str):
        """Initialize agent message.

        Args:
            content: Message content.
        """
        super().__init__("assistant", content)


class AgentContext:
    """
    Conversation context that can be copied (independent history) or shared
    (multiple agents modify same history) between agents.
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
    ):
        """Initialize agent context.

        Args:
            name: Context name.
            system_prompt: System prompt for this context.
        """
        self.name = name
        self.system_prompt = Msg(role="system", content=system_prompt or "")
        self.message_history: list[dict[str, str]] = []
        self.completed = False

    def add_message(self, msg: Msg) -> None:
        """Add a message to the history."""
        self.message_history.append(msg.raw())

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt["content"]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt["content"] = system_prompt

    def get_last_output(self) -> str:
        """Get the last assistant message."""
        for i in range(len(self.message_history) - 1, -1, -1):
            if self.message_history[i]["role"] == "assistant":
                return self.message_history[i]["content"]
        return ""

    def get_last_input(self) -> str:
        """Get the last user message."""
        for i in range(len(self.message_history) - 1, -1, -1):
            if self.message_history[i]["role"] == "user":
                return self.message_history[i]["content"]
        return ""

    def clear(self) -> None:
        """Clear all message history."""
        self.message_history.clear()

    def remove_last_message(self) -> None:
        """Remove the last message from history."""
        if self.message_history:
            self.message_history.pop()

    def copy(self, new_name: Optional[str] = None) -> "AgentContext":
        """
        Create an independent copy of this context.
        Changes to the copy will not affect the original.
        """
        new_ctx = AgentContext(
            new_name or f"{self.name}_copy",
            self.system_prompt["content"],
        )
        new_ctx.message_history = deepcopy(self.message_history)
        new_ctx.completed = self.completed
        return new_ctx

    def get_messages(self, workspace: Optional[str] = None) -> list[dict[str, str]]:
        """Get all messages including system prompt and optional workspace.

        Internal ``_pithos_*`` metadata keys are stripped from each message
        before the list is returned so the LLM backend never receives them.
        """
        messages = []
        if self.system_prompt["content"]:
            messages.append(self.system_prompt.raw())
        if workspace:
            messages.append({"role": "user", "content": workspace})
        for msg in self.message_history:
            # Strip internal pithos metadata so the LLM only sees role+content
            clean = {k: v for k, v in msg.items() if not k.startswith("_pithos_")}
            messages.append(clean)
        return messages

    def to_dict(self, with_history: bool = False) -> dict[str, Any]:
        """Serialize context to dictionary."""
        d: dict[str, Any] = {"system_prompt": self.system_prompt["content"]}
        if with_history:
            d["message_history"] = self.message_history
        return d

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        name: str = "default",
        config_manager: Optional[ConfigManager] = None,
    ) -> "AgentContext":
        """Deserialize context from dictionary."""
        ctx = cls(name, data.get("system_prompt", ""))
        if "message_history" in data:
            ctx.message_history = data["message_history"]
        return ctx
