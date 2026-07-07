"""
backend/api_models.py

Pydantic request/response models for the dashboard API.

These are the HTTP-facing shapes returned to the React frontend. They
mirror `schemas.ToolExecutionResult` (the framework-agnostic dataclass
used internally by the agents) but are defined separately so the core
agent layer never has to depend on Pydantic/FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ChartType = Literal["bar", "pie", "line"]
SessionCategory = Literal["chat", "sql", "research"]


class ChartSpecModel(BaseModel):
    chart_type: ChartType
    title: str
    labels: List[str]
    values: List[float]


class ChatRequest(BaseModel):
    """Body for POST /api/chat."""

    message: str = Field(..., min_length=1, description="The user's question.")
    session_id: Optional[str] = Field(
        None,
        description="Existing session to continue. If omitted, a new session is created.",
    )


class ChatMessageModel(BaseModel):
    """A single message (user or assistant) as stored/returned for a session."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    tool_used: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    generated_sql: Optional[str] = None
    table_data: Optional[List[Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, str]]] = None
    chart: Optional[ChartSpecModel] = None
    created_at: datetime


class ChatResponse(BaseModel):
    """Response for POST /api/chat."""

    session_id: str
    message: ChatMessageModel


class SessionSummary(BaseModel):
    """A row in the sidebar's chat/SQL/research history lists."""

    id: str
    title: str
    category: SessionCategory
    saved: bool
    created_at: datetime
    updated_at: datetime
    message_count: int


class SessionDetail(BaseModel):
    """Full session detail, including all messages."""

    id: str
    title: str
    category: SessionCategory
    saved: bool
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageModel]


class CreateSessionResponse(BaseModel):
    session_id: str


class ExportRequest(BaseModel):
    """Body for POST /api/export/{format}."""

    table_data: List[Dict[str, Any]] = Field(..., min_length=1)
    filename: str = Field("export", description="Base filename, without extension.")
