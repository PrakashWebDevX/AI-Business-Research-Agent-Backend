"""
backend/main.py

FastAPI application for the AI Business Research Agent dashboard.

Responsibilities:
    - Expose the existing `BusinessAgent` (agent.py) over HTTP so the
      React frontend can send chat messages and receive structured
      results (answer, tool used, execution time, generated SQL, table
      data, chart, sources).
    - Manage chat sessions (create, list by sidebar category, fetch,
      save/unsave, delete) via backend/session_store.py.
    - Export SQL result tables as CSV, Excel, or JSON.

Run with:
    uvicorn backend.main:app --reload --port 8000

Design note — a single shared BusinessAgent:
    This backend keeps ONE `BusinessAgent` instance for the whole
    process (created lazily on first request) rather than one per
    session. Each session's own message history is still stored
    correctly and independently in `session_store`; when a request
    switches to a different session than the one most recently active,
    the backend calls `agent.load_history(...)` to resync the agent's
    live conversational memory to that session before answering. This
    gives correct, isolated multi-session behavior for a single active
    user (e.g. switching between saved chats), while keeping the
    implementation simple. It is NOT safe for true concurrent multi-user
    traffic (two people chatting at the same instant would contend for
    the same agent) — see the README for notes on extending this to a
    per-session or per-user agent pool if that's ever needed.
"""



from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import rag

from fastapi.responses import Response

from backend import session_store
from backend.api_models import (
    ChartSpecModel,
    ChatMessageModel,
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    ExportRequest,
    SessionDetail,
    SessionSummary,
)
from backend.export_utils import EXPORT_FORMATS
from backend.session_store import Session, StoredMessage

logger = logging.getLogger(__name__)

# One-time diagnostic: confirm exactly which langchain versions are
# actually installed in this deployment. Safe to remove once the
# __arg1 tool-schema issue is confirmed resolved.
import langchain_core
import langchain

logger.info("langchain version: %s", langchain.__version__)
logger.info("langchain_core version: %s", langchain_core.__version__)

rag.init_rag_tables()

app = FastAPI(
    title="AI Business Research Agent API",
    description="Backend API powering the AI Business Research Agent dashboard.",
    version="1.0.0",
)

# Allow the local Vite dev server (and a same-origin production build)
# to call this API. Adjust origins here if you deploy the frontend
# somewhere other than localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-business-research-agent.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Shared agent (lazy singleton — see module docstring)
# --------------------------------------------------------------------------- #

_agent = None  # type: ignore[var-annotated]
_active_session_id: Optional[str] = None


def _get_agent():
    """Return the shared BusinessAgent, creating it on first use."""
    global _agent
    if _agent is None:
        # Imported lazily so the API can start up (e.g. for `/api/health`
        # or interactive docs) even before GROQ_API_KEY is configured;
        # the clear EnvironmentError only surfaces when chat is first used.
        from agent import BusinessAgent

        logger.info("Lazily initializing BusinessAgent for the API...")
        _agent = BusinessAgent(verbose=False)
    return _agent


def _ensure_session_is_active(session: Session) -> None:
    """
    Make sure the shared agent's live conversational memory reflects the
    given session before it answers. If a different session was active
    most recently, resync the agent's history from this session's
    stored messages (see module docstring for why this is needed).
    """
    global _active_session_id
    agent = _get_agent()
    if _active_session_id != session.id:
        agent.load_history(session_store.get_history_for_agent(session))
        _active_session_id = session.id


def _stored_message_to_model(message: StoredMessage) -> ChatMessageModel:
    chart_model = ChartSpecModel(**message.chart) if message.chart else None
    return ChatMessageModel(
        id=message.id,
        role=message.role,
        content=message.content,
        tool_used=message.tool_used,
        execution_time_seconds=message.execution_time_seconds,
        generated_sql=message.generated_sql,
        table_data=message.table_data,
        sources=message.sources,
        chart=chart_model,
        created_at=message.created_at,
    )


def _session_to_summary(session: Session) -> SessionSummary:
    return SessionSummary(
        id=session.id,
        title=session.title,
        category=session.category,
        saved=session.saved,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    """Basic liveness check — does not require API keys to succeed."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = rag.add_document(file.filename, file_bytes)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/documents")
def get_documents():
    return rag.list_documents()


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: int):
    deleted = rag.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session = session_store.get_or_create_session(request.session_id)

    try:
        _ensure_session_is_active(session)
        session_store.add_user_message(session, request.message)
        agent = _get_agent()
        result = agent.ask_structured(request.message)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a raw 500 to the frontend
        logger.exception("Unhandled error in /api/chat")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

    assistant_message = session_store.add_assistant_message(session, result)

    return ChatResponse(
        session_id=session.id,
        message=_stored_message_to_model(assistant_message),
    )


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    """Create a new, empty chat session (used by the "New Chat" button)."""
    session = session_store.create_session()
    return CreateSessionResponse(session_id=session.id)


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions(category: Optional[str] = None, saved_only: bool = False) -> list[SessionSummary]:
    """
    List sessions for the sidebar.

    Args:
        category: Optional filter — "chat", "sql", or "research" — for
            the sidebar's "Chat History" / "SQL Queries" / "Research
            History" sections.
        saved_only: If true, return only sessions marked as saved (for
            the "Saved Reports" section), ignoring `category`.
    """
    if saved_only:
        sessions = session_store.list_saved_sessions()
    elif category:
        if category not in ("chat", "sql", "research"):
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        sessions = session_store.list_sessions_by_category(category)  # type: ignore[arg-type]
    else:
        sessions = session_store.list_sessions()

    return [_session_to_summary(s) for s in sessions]


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> SessionDetail:
    """Fetch full detail (all messages) for one session."""
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDetail(
        id=session.id,
        title=session.title,
        category=session.category,
        saved=session.saved,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[_stored_message_to_model(m) for m in session.messages],
    )


@app.post("/api/sessions/{session_id}/save", response_model=SessionSummary)
def save_session(session_id: str, saved: bool = True) -> SessionSummary:
    """Mark a session as saved (or unsaved), for the "Saved Reports" sidebar section."""
    session = session_store.set_saved(session_id, saved)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_summary(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a session."""
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

@app.post("/api/export/{export_format}")
def export_table(export_format: str, request: ExportRequest) -> Response:
    """
    Export a SQL result table (as displayed in the dashboard) to a
    downloadable file.

    Args:
        export_format: One of "csv", "excel", or "json".
    """
    if export_format not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format '{export_format}'. Use one of: {list(EXPORT_FORMATS)}",
        )

    builder, media_type, extension = EXPORT_FORMATS[export_format]
    file_bytes = builder(request.table_data)
    filename = f"{request.filename}.{extension}"

    return Response(
        content=file_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )