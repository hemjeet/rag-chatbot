"""
Session-based conversation memory manager.
Each session (browser tab / API client) gets its own isolated memory buffer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config import settings


@dataclass
class Message:
    """A single chat message."""
    role: str          # "user" or "assistant"
    content: str


@dataclass
class Session:
    """Holds the conversation history for one session."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    max_window: int = field(default_factory=lambda: settings.memory_window)

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self):
        """Keep only the last `max_window * 2` messages (each exchange = 2 msgs)."""
        max_msgs = self.max_window * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def get_history(self) -> List[Dict[str, str]]:
        """Return history as a list of dicts for serialization / prompt injection."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_history_as_text(self) -> str:
        """Format history as text for injection into the RAG system prompt."""
        if not self.messages:
            return ""
        lines = []
        for m in self.messages:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def clear(self):
        self.messages.clear()


class MemoryManager:
    """
    Manages multiple conversation sessions in-memory.
    Thread-safe enough for the single-process Gradio+FastAPI setup.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get an existing session or create a new one."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, returns None if not found."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())


# Singleton memory manager
memory_manager = MemoryManager()
