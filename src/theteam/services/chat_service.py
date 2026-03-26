"""
Chat service - manages conversations and message exchange.
"""

import logging
import uuid
import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a chat message."""

    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    """Represents a conversation."""

    id: str
    title: str
    agent_id: Optional[str]
    created_at: str
    updated_at: str
    messages: list[Message]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
        }


class ChatService:
    """Service for managing chat conversations."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize chat service.

        Args:
            storage_dir: Directory for storing conversation files.
                        Defaults to data/conversations in current working directory.
        """
        if storage_dir is None:
            storage_dir = Path.cwd() / "data" / "conversations"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory conversations
        self.conversations: dict[str, Conversation] = {}
        self._load_conversations()

        self.lock = threading.Lock()

    def _load_conversations(self):
        """Load conversations from disk."""
        for conv_file in self.storage_dir.glob("*.json"):
            try:
                with open(conv_file, "r") as f:
                    data = json.load(f)

                messages = [Message(**msg) for msg in data["messages"]]
                conversation = Conversation(
                    id=data["id"],
                    title=data["title"],
                    agent_id=data.get("agent_id"),
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    messages=messages,
                )
                self.conversations[conversation.id] = conversation

            except Exception as e:
                logger.error(f"Error loading conversation from {conv_file}: {e}")

    def _save_conversation(self, conversation: Conversation):
        """Save a conversation to disk."""
        conv_file = self.storage_dir / f"{conversation.id}.json"
        try:
            with open(conv_file, "w") as f:
                json.dump(conversation.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {e}")
            raise

    def list_conversations(self) -> list[dict]:
        """List all conversations.

        Returns:
            List of conversation summaries sorted by updated_at (most recent first).
        """
        with self.lock:
            conversations = []
            for conv in self.conversations.values():
                conversations.append(
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "agent_id": conv.agent_id,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                        "message_count": len(conv.messages),
                    }
                )

            # Sort by updated_at (most recent first)
            conversations.sort(key=lambda x: x["updated_at"], reverse=True)
            return conversations

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a specific conversation with all messages.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation dictionary or None if not found.
        """
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                return None
            return conversation.to_dict()

    def create_conversation(
        self, agent_id: Optional[str] = None, title: Optional[str] = None
    ) -> str:
        """Create a new conversation.

        Args:
            agent_id: Optional agent identifier for this conversation.
            title: Optional conversation title.

        Returns:
            Created conversation ID.
        """
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        conversation = Conversation(
            id=conversation_id,
            title=title,
            agent_id=agent_id,
            created_at=timestamp,
            updated_at=timestamp,
            messages=[],
        )

        with self.lock:
            self.conversations[conversation_id] = conversation
            self._save_conversation(conversation)

        logger.info(f"Created conversation {conversation_id}")
        return conversation_id

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if deletion successful, False if conversation not found.
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return False

            del self.conversations[conversation_id]

            # Delete from disk
            conv_file = self.storage_dir / f"{conversation_id}.json"
            if conv_file.exists():
                conv_file.unlink()

            logger.info(f"Deleted conversation {conversation_id}")
            return True

    def send_message(
        self,
        conversation_id: str,
        message: str,
        client_id: Optional[str] = None,
        socketio=None,
        stream: bool = False,
    ) -> str:
        """Send a message and get agent response (async).

        Args:
            conversation_id: Conversation identifier.
            message: User message content.
            client_id: Optional client identifier for socket events.
            socketio: Optional socket.io instance for real-time updates.
            stream: Whether to stream the response token by token via SocketIO.

        Returns:
            Message ID of the sent user message.

        Raises:
            ValueError: If conversation not found.
        """
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Add user message
            message_id = str(uuid.uuid4())
            user_message = Message(
                id=message_id,
                role="user",
                content=message,
                timestamp=datetime.now().isoformat(),
            )
            conversation.messages.append(user_message)
            conversation.updated_at = datetime.now().isoformat()
            self._save_conversation(conversation)

        # Start async processing
        target = self._process_message_streaming if stream else self._process_message
        thread = threading.Thread(
            target=target,
            args=(conversation_id, message, socketio, client_id),
        )
        thread.daemon = True
        thread.start()

        return message_id

    def _process_message(
        self, conversation_id: str, message: str, socketio, client_id: Optional[str]
    ):
        """Process message and generate response (runs in background thread).

        Args:
            conversation_id: Conversation identifier.
            message: User message content.
            socketio: Socket.io instance for real-time updates.
            client_id: Optional client identifier for socket events.
        """
        from pithos.agent import OllamaAgent
        from theteam.services.agent_service import AgentService
        from theteam.api.socketio_handlers import emit_to_client

        try:
            with self.lock:
                conversation = self.conversations.get(conversation_id)
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")

                agent_id = conversation.agent_id

            # Get agent configuration
            agent_service = AgentService()
            agent_config = agent_service.get_agent(agent_id) if agent_id else None

            if not agent_config:
                # Use default agent
                logger.warning(
                    f"No agent specified for conversation {conversation_id}, using default"
                )
                agent_config = {
                    "config": {"model": "glm-4.7-flash:latest", "name": "Default Agent"}
                }

            # Notify that processing started
            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "message_processing",
                    {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            # Create agent and generate response
            config = agent_config["config"]
            model = (
                config.get("default_model")
                or config.get("model")
                or "glm-4.7-flash:latest"
            )
            agent = OllamaAgent(
                default_model=model,
                system_prompt=config.get("system_prompt") or "",
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
            )

            # Build context from conversation history
            context = []
            with self.lock:
                conversation = self.conversations[conversation_id]
                for msg in conversation.messages:
                    context.append(f"{msg.role}: {msg.content}")

            # Generate response
            response_text = agent.send(message)

            # Add assistant message
            with self.lock:
                conversation = self.conversations[conversation_id]
                assistant_message = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now().isoformat(),
                )
                conversation.messages.append(assistant_message)
                conversation.updated_at = datetime.now().isoformat()
                self._save_conversation(conversation)

            # Send response to client
            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "message_response",
                    {
                        "conversation_id": conversation_id,
                        "message": assistant_message.to_dict(),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(
                f"Error processing message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "message_error",
                    {
                        "conversation_id": conversation_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

    def _process_message_streaming(
        self, conversation_id: str, message: str, socketio, client_id: Optional[str]
    ):
        """Process message and stream the response token by token (runs in background thread).

        Emits the following SocketIO events to the client:
        - ``message_processing``: fired immediately when processing starts.
        - ``stream_start``: fired before the first token, carries the new ``message_id``.
        - ``stream_chunk``: fired for every token chunk with fields
          ``{conversation_id, message_id, chunk}``.
        - ``stream_end``: fired when streaming is complete, carries the full
          ``message`` dict so the client can replace the in-progress placeholder.
        - ``message_error``: fired on failure.

        Args:
            conversation_id: Conversation identifier.
            message: User message content.
            socketio: Socket.io instance for real-time updates.
            client_id: Optional client identifier for socket events.
        """
        from pithos.agent import OllamaAgent
        from theteam.services.agent_service import AgentService
        from theteam.api.socketio_handlers import emit_to_client

        try:
            with self.lock:
                conversation = self.conversations.get(conversation_id)
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                agent_id = conversation.agent_id

            agent_service = AgentService()
            agent_config = agent_service.get_agent(agent_id) if agent_id else None

            if not agent_config:
                logger.warning(
                    f"No agent specified for conversation {conversation_id}, using default"
                )
                agent_config = {
                    "config": {"model": "glm-4.7-flash:latest", "name": "Default Agent"}
                }

            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "message_processing",
                    {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            config = agent_config["config"]
            model = (
                config.get("default_model")
                or config.get("model")
                or "glm-4.7-flash:latest"
            )
            agent = OllamaAgent(
                default_model=model,
                system_prompt=config.get("system_prompt") or "",
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
            )

            # Pre-allocate the ID that will identify the streaming message
            assistant_message_id = str(uuid.uuid4())

            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "stream_start",
                    {
                        "conversation_id": conversation_id,
                        "message_id": assistant_message_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            # Consume the stream, forwarding every chunk to the client
            full_response = ""
            for chunk in agent.stream(message):
                full_response += chunk
                if socketio and client_id:
                    emit_to_client(
                        socketio,
                        client_id,
                        "stream_chunk",
                        {
                            "conversation_id": conversation_id,
                            "message_id": assistant_message_id,
                            "chunk": chunk,
                        },
                    )

            # Persist the completed message
            with self.lock:
                conversation = self.conversations[conversation_id]
                assistant_message = Message(
                    id=assistant_message_id,
                    role="assistant",
                    content=full_response,
                    timestamp=datetime.now().isoformat(),
                )
                conversation.messages.append(assistant_message)
                conversation.updated_at = datetime.now().isoformat()
                self._save_conversation(conversation)

            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "stream_end",
                    {
                        "conversation_id": conversation_id,
                        "message": assistant_message.to_dict(),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(
                f"Error streaming message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            if socketio and client_id:
                emit_to_client(
                    socketio,
                    client_id,
                    "message_error",
                    {
                        "conversation_id": conversation_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

    def update_conversation_agent(self, conversation_id: str, agent_id: str) -> bool:
        """Update the agent for a conversation.

        Args:
            conversation_id: Conversation identifier.
            agent_id: New agent identifier.

        Returns:
            True if update successful, False if conversation not found.
        """
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                return False

            conversation.agent_id = agent_id
            conversation.updated_at = datetime.now().isoformat()
            self._save_conversation(conversation)

            logger.info(
                f"Updated agent for conversation {conversation_id} to {agent_id}"
            )
            return True

    def add_system_message(self, conversation_id: str, message: str) -> bool:
        """Add a system message to a conversation.

        Args:
            conversation_id: Conversation identifier.
            message: System message content.

        Returns:
            True if message added successfully, False if conversation not found.
        """
        with self.lock:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                return False

            system_message = Message(
                id=str(uuid.uuid4()),
                role="system",
                content=message,
                timestamp=datetime.now().isoformat(),
            )
            conversation.messages.append(system_message)
            conversation.updated_at = datetime.now().isoformat()
            self._save_conversation(conversation)

            return True
