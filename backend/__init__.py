"""
backend package

The FastAPI web layer for the AI Business Research Agent dashboard.

This package wraps the existing agent (agent.BusinessAgent) with an
HTTP API so the React frontend (frontend/) can drive it: sending chat
messages, browsing session history, and exporting SQL results as
CSV/Excel/JSON.

It intentionally contains no agent logic of its own — every question
is still answered by the same `BusinessAgent` used by the CLI in
app.py. This package only adds session management, request/response
shaping, and file export on top.
"""
