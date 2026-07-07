"""
backend/session_store.py

A simple, in-memory store for chat sessions.

Responsibilities:
    - Create and track chat sessions (each a sequence of user/assistant
      messages), so the dashboard's sidebar can show "Chat History",
      "SQL Queries", "Research History", and "Saved Reports".
    - Categorize each session based on the dominant tool used across its
      assistant messages (SQL Agent -> "sql", Web Research -> "research",
      Mixed/Direct Answer -> "chat"), so it appears in the right sidebar
      list without the frontend needing to compute that itself.
    - Derive a human-readable title from the session's first user message.

This is intentionally in-memory (a Python dict), not backed by a
database — sessions are lost on server restart. That's a reasonable
trade-off for a portfolio/demo dashboard; swapping in a persistent
store (e.g. a `sessions` table in the existing SQLite database) later
would only require changing this one file, since `backend/main.py`
only calls the functions defined here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from schemas import ToolExecutionResult

SessionCategory = Literal["chat", "sql", "research"]

_TITLE_MAX_LENGTH = 60


@dataclass
class StoredMessage:
    id: str
    role: Literal["user", "assistant"]
    content: str
    tool_used: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    generated_sql: Optional[str] = None
    table_data: Optional[list] = None
    sources: Optional[list] = None
    chart: Optional[dict] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Session:
    id: str
    title: str
    category: SessionCategory
    saved: bool
    created_at: datetime
    updated_at: datetime
    messages: List[StoredMessage] = field(default_factory=list)


# Process-wide in-memory store. See module docstring for the rationale
# and the upgrade path to persistent storage.
_sessions: Dict[str, Session] = {}


def create_session() -> Session:
    """Create and store a new, empty chat session."""
    now = datetime.now(timezone.utc)
    session_id = str(uuid.uuid4())
    session = Session(
        id=session_id,
        title="New Chat",
        category="chat",
        saved=False,
        created_at=now,
        updated_at=now,
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    """Return the session with the given id, or None if it doesn't exist."""
    return _sessions.get(session_id)


def get_or_create_session(session_id: Optional[str]) -> Session:
    """
    Return the session with the given id if it exists, otherwise create
    a brand-new session. Used by POST /api/chat, which lets the frontend
    omit `session_id` to start a new conversation implicitly.
    """
    if session_id:
        existing = get_session(session_id)
        if existing:
            return existing
    return create_session()


def list_sessions() -> List[Session]:
    """Return all sessions, most recently updated first."""
    return sorted(_sessions.values(), key=lambda s: s.updated_at, reverse=True)


def list_sessions_by_category(category: SessionCategory) -> List[Session]:
    """Return sessions in a given sidebar category, most recently updated first."""
    return [s for s in list_sessions() if s.category == category]


def list_saved_sessions() -> List[Session]:
    """Return sessions the user has explicitly saved ("Saved Reports")."""
    return [s for s in list_sessions() if s.saved]


def set_saved(session_id: str, saved: bool) -> Optional[Session]:
    """Mark a session as saved/unsaved. Returns the updated session, or None if not found."""
    session = get_session(session_id)
    if session is None:
        return None
    session.saved = saved
    session.updated_at = datetime.now(timezone.utc)
    return session


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed and was removed."""
    return _sessions.pop(session_id, None) is not None


def add_user_message(session: Session, content: str) -> StoredMessage:
    """Append a user message to a session, updating its title if this is the first message."""
    message = StoredMessage(id=str(uuid.uuid4()), role="user", content=content)
    session.messages.append(message)
    if len(session.messages) == 1:
        session.title = (
            content if len(content) <= _TITLE_MAX_LENGTH else content[: _TITLE_MAX_LENGTH - 1] + "…"
        )
    session.updated_at = datetime.now(timezone.utc)
    return message


def add_assistant_message(session: Session, result: ToolExecutionResult) -> StoredMessage:
    """
    Append an assistant message to a session, storing the full
    structured result (tool used, timing, SQL, table data, chart,
    sources), and re-categorize the session based on which tool ran.
    """
    chart_dict = None
    if result.chart is not None:
        chart_dict = {
            "chart_type": result.chart.chart_type,
            "title": result.chart.title,
            "labels": result.chart.labels,
            "values": result.chart.values,
        }

    message = StoredMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=result.answer,
        tool_used=result.tool_used,
        execution_time_seconds=result.execution_time_seconds,
        generated_sql=result.generated_sql,
        table_data=result.table_data,
        sources=result.sources,
        chart=chart_dict,
    )
    session.messages.append(message)
    session.category = _infer_category(result.tool_used)
    session.updated_at = datetime.now(timezone.utc)
    return message


def get_history_for_agent(session: Session) -> List[dict]:
    """
    Return this session's messages as plain {"role", "content"} dicts,
    suitable for `BusinessAgent.load_history()`.
    """
    return [{"role": m.role, "content": m.content} for m in session.messages]


def _infer_category(tool_used: str) -> SessionCategory:
    """Map a ToolExecutionResult.tool_used label to a sidebar category."""
    if tool_used == "SQL Agent":
        return "sql"
    if tool_used == "Web Research":
        return "research"
    return "chat"  # "Mixed" and "Direct Answer" both live in general chat history.
