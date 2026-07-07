"""
utils package

Shared, dependency-free helper functions used across multiple modules
in the AI Business Research Agent (agent.py, sql_agent.py, web_agent.py).

Keeping these here — rather than duplicating small helpers in each
agent module — means a fix or improvement only has to be made once.
"""

from .output_parsing import extract_output_text

__all__ = ["extract_output_text"]
