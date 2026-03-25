"""Message-based data routing for flowcharts."""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Message:
    """A message carrying data between nodes."""

    data: Any
    """The message payload."""

    source_node: Optional[str] = None
    """The node that produced this message."""

    target_node: Optional[str] = None
    """The node that should receive this message."""

    input_key: str = "default"
    """The input key/port this message is for."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this message."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this message was created."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for this message."""

    def __repr__(self) -> str:
        return f"Message({self.source_node} -> {self.target_node}:{self.input_key}, id={self.message_id[:8]})"


@dataclass
class NodeInputState:
    """Tracks the input state for a node."""

    node_id: str
    """The node this state belongs to."""

    required_inputs: list[str]
    """List of required input keys."""

    received_inputs: dict[str, Message] = field(default_factory=dict)
    """Map of input_key to received Message."""

    optional_inputs: list[str] = field(default_factory=list)
    """List of optional input keys."""

    def is_ready(self) -> bool:
        """Check if all required inputs have been received."""
        return all(key in self.received_inputs for key in self.required_inputs)

    def receive_message(self, message: Message) -> None:
        """Receive a message for this node.

        Args:
            message: The message to receive.

        Raises:
            ValueError: If message is None.
        """
        if message is None:
            raise ValueError("message cannot be None")
        self.received_inputs[message.input_key] = message

    def get_input_data(self, key: str = "default") -> Any:
        """Get the data from a specific input."""
        if key in self.received_inputs:
            return self.received_inputs[key].data
        return None

    def get_all_input_data(self) -> dict[str, Any]:
        """Get all input data as a dictionary."""
        return {key: msg.data for key, msg in self.received_inputs.items()}

    def reset(self) -> None:
        """Clear all received inputs."""
        self.received_inputs.clear()


class MessageRouter:
    """Routes messages between nodes in a flowchart."""

    def __init__(self, max_history: int = 0):
        """Initialize the message router.

        Args:
            max_history: Maximum number of messages to keep in ``message_history``.
                         The oldest messages are evicted when the limit is exceeded,
                         producing a rolling window.  ``0`` means unlimited (default).
        """
        self.pending_messages: list[Message] = []
        self.node_states: dict[str, NodeInputState] = {}
        self.message_history: list[Message] = []
        self.shared_context: dict[str, Any] = {}  # Shared context for all nodes
        self._max_history: int = max_history

    def register_node(
        self,
        node_id: str,
        required_inputs: list[str],
        optional_inputs: Optional[list[str]] = None,
    ) -> None:
        """Register a node with its input requirements.

        Args:
            node_id: ID of the node to register.
            required_inputs: List of required input keys.
            optional_inputs: Optional list of optional input keys.

        Raises:
            ValueError: If node_id is empty or required_inputs is None/empty.
        """
        if not node_id or not node_id.strip():
            raise ValueError("node_id cannot be empty")
        if not required_inputs:
            raise ValueError("required_inputs cannot be None or empty")

        self.node_states[node_id] = NodeInputState(
            node_id=node_id,
            required_inputs=required_inputs,
            optional_inputs=optional_inputs or [],
        )

    def send_message(self, message: Message) -> None:
        """Send a message to a target node.

        Args:
            message: The message to send.

        Raises:
            ValueError: If message is None.
        """
        if message is None:
            raise ValueError("message cannot be None")

        self.pending_messages.append(message)
        self.message_history.append(message)

        # Enforce rolling window — drop oldest entries when over the limit.
        if self._max_history > 0 and len(self.message_history) > self._max_history:
            del self.message_history[: len(self.message_history) - self._max_history]

        # If target node exists, deliver the message
        if message.target_node and message.target_node in self.node_states:
            self.node_states[message.target_node].receive_message(message)

    def get_ready_nodes(self) -> list[str]:
        """Get list of nodes that have all required inputs."""
        ready = []
        for node_id, state in self.node_states.items():
            if state.is_ready():
                ready.append(node_id)
        return ready

    def get_node_state(self, node_id: str) -> Optional[NodeInputState]:
        """Get the input state for a specific node."""
        return self.node_states.get(node_id)

    def clear_node_inputs(self, node_id: str) -> None:
        """Clear inputs for a node after it has executed."""
        if node_id in self.node_states:
            self.node_states[node_id].reset()

    def reset(self) -> None:
        """Reset the router to initial state."""
        self.pending_messages.clear()
        self.message_history.clear()
        for state in self.node_states.values():
            state.reset()
        # Note: shared_context is NOT cleared on reset to preserve injected dependencies
