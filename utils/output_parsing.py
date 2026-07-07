"""
utils/output_parsing.py

Shared helper for normalizing LangChain AgentExecutor output into a
plain string.

Why this exists:
    Depending on the LLM provider and whether streaming/tool-calling is
    involved, `AgentExecutor.invoke(...)["output"]` is not always a
    plain string. Some providers (this was originally observed with
    Gemini via `ChatGoogleGenerativeAI`, before this project switched
    to Groq) can return message content as a list of content blocks,
    e.g.:

        [{"type": "text", "text": "The answer is 42."}]

    instead of the plain string:

        "The answer is 42."

    Calling `.strip()` directly on that list raises:
        AttributeError: 'list' object has no attribute 'strip'

    This module provides a single, well-tested function that safely
    flattens any of these shapes into a plain string, so every agent in
    the project (agent.py, sql_agent.py, web_agent.py) can rely on
    `extract_output_text(...)` instead of duplicating ad hoc handling.
"""

from __future__ import annotations

from typing import Any


def extract_output_text(raw_output: Any) -> str:
    """
    Normalize an AgentExecutor's `output` field into a plain string.

    Handles every shape LangChain is known to return for `output`:
        - A plain string -> returned as-is.
        - A list of content blocks, where each block is either:
            - a string, or
            - a dict with a "text" key (e.g. {"type": "text", "text": "..."}).
          -> all text pieces are concatenated in order.
        - None -> returns an empty string.
        - Anything else -> falls back to `str(raw_output)`.

    Args:
        raw_output: The raw value of `result["output"]` from an
            `AgentExecutor.invoke(...)` call (or any similarly-shaped
            LangChain response), whose exact type can vary by LLM
            provider and by whether tool-calling/streaming was used.

    Returns:
        A plain string containing the concatenated text content. Does
        NOT strip whitespace — callers should call `.strip()` on the
        result themselves if needed, so this function stays focused on
        one responsibility: type normalization.

    Examples:
        >>> extract_output_text("Hello world")
        'Hello world'
        >>> extract_output_text([{"type": "text", "text": "Hello world"}])
        'Hello world'
        >>> extract_output_text([{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}])
        'Hello world'
        >>> extract_output_text(None)
        ''
    """
    if raw_output is None:
        return ""

    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        pieces: list[str] = []
        for item in raw_output:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                # Common LangChain content-block shapes:
                #   {"type": "text", "text": "..."}
                #   {"text": "..."}
                text_value = item.get("text")
                if isinstance(text_value, str):
                    pieces.append(text_value)
            else:
                pieces.append(str(item))
        return "".join(pieces)

    # Unknown shape (e.g. an unexpected object) — fall back safely
    # rather than raising, since this function sits on the critical
    # path of every agent's response handling.
    return str(raw_output)
