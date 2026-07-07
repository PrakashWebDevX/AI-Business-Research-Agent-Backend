"""
schemas.py

Shared, framework-agnostic data structures used across the project.

Responsibilities:
    - Define `ToolExecutionResult`, the structured result returned by
      SQLAgent, WebAgent, and BusinessAgent whenever a caller needs more
      than just the final text answer — e.g. the FastAPI backend, which
      needs to know which tool ran, how long it took, the generated SQL,
      tabular data for tables/exports, chart data, and web sources.
    - Define `ChartSpec`, a small, chart-library-agnostic description of
      a chart the frontend can render (bar / pie / line).

These are plain `dataclasses`, not Pydantic models, so that the core
agent layer (agent.py, sql_agent.py, web_agent.py) has no dependency on
FastAPI or Pydantic. The backend layer converts these to JSON via
`dataclasses.asdict()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

ChartType = Literal["bar", "pie", "line"]


@dataclass
class ChartSpec:
    """A minimal, frontend-agnostic chart description."""

    chart_type: ChartType
    title: str
    labels: List[str]
    values: List[float]


@dataclass
class ToolExecutionResult:
    """
    The structured result of answering one question, whether it came
    from SQLAgent, WebAgent, or the top-level BusinessAgent.

    Attributes:
        answer: The final, clean natural-language answer.
        tool_used: A human-readable label — "SQL Agent", "Web Research",
            "Mixed", or "Direct Answer" — describing what produced the
            answer. Intended for direct display in the UI.
        execution_time_seconds: Wall-clock time taken to produce the
            answer, rounded to 2 decimal places.
        generated_sql: The read-only SQL query the SQL agent executed,
            if applicable.
        table_data: Query result rows as a list of plain dicts
            (column_name -> value), suitable for rendering a table or
            exporting to CSV/Excel/JSON.
        sources: Web sources used, as a list of {"title": ..., "url": ...}
            dicts, if applicable.
        chart: An auto-generated chart suggestion, if the result data
            was suitable for visualization.
    """

    answer: str
    tool_used: str
    execution_time_seconds: float
    generated_sql: Optional[str] = None
    table_data: Optional[List[Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, str]]] = None
    chart: Optional[ChartSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain, JSON-serializable dict (for the API layer)."""
        return asdict(self)
