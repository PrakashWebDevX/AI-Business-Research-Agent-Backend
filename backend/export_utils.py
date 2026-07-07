"""
backend/export_utils.py

Helpers for exporting SQL query result rows (a list of plain dicts) to
CSV, Excel (.xlsx), or JSON, for the dashboard's "Export CSV / Export
Excel / Download JSON" buttons.

Uses pandas + openpyxl for CSV/Excel so column types and formatting are
handled correctly with minimal code; JSON export is just a direct dump
since the rows are already plain dicts.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Tuple

import pandas as pd


def to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert result rows to CSV file bytes."""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert result rows to an .xlsx file's bytes."""
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def to_json_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert result rows to pretty-printed JSON file bytes."""
    return json.dumps(rows, indent=2, default=str).encode("utf-8")


# Maps a format string (from the URL path) to (bytes-builder, media type, file extension).
EXPORT_FORMATS: Dict[str, Tuple[Any, str, str]] = {
    "csv": (to_csv_bytes, "text/csv", "csv"),
    "excel": (
        to_excel_bytes,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
    ),
    "json": (to_json_bytes, "application/json", "json"),
}
