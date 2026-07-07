"""
tools.py

Interface-adapter layer for the AI Business Research Agent.

Responsibilities:
    - Expose the project's two capabilities — querying the internal
      database and searching the web — as standard LangChain `Tool`
      objects.
    - Act as a single registry that the orchestrator (agent.py) imports
      from, rather than reaching into sql_agent.py / web_agent.py
      directly. This keeps agent.py decoupled from *how* each tool is
      implemented.

Design notes:
    - Each tool is built by its own small, focused function
      (`build_sql_tool`, `build_web_search_tool`), so tools can be
      added, removed, or swapped independently — this is what keeps the
      file modular.
    - The underlying agents (SQLAgent, WebAgent) are created lazily, on
      first use, not at import time. This means importing tools.py never
      fails just because an API key isn't set yet; the error only
      surfaces if that specific tool is actually invoked.
    - `get_all_tools()` is the single function the orchestrator should
      call to get the full toolset. Adding a new tool later only means
      writing a new `build_*_tool()` function and adding it to that list.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import Tool

from sql_agent import SQLAgent
from web_agent import WebAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Lazy singletons
# --------------------------------------------------------------------------- #
# Both agents are expensive to construct (they spin up an LLM client and,
# in SQLAgent's case, initialize/seed the database) and hold state that's
# useful to reuse across calls (e.g. WebAgent's conversation memory). They
# are therefore created once, on first use, and cached here.

_sql_agent_instance: Optional[SQLAgent] = None
_web_agent_instance: Optional[WebAgent] = None


def _get_sql_agent() -> SQLAgent:
    """Return the shared SQLAgent instance, creating it on first use."""
    global _sql_agent_instance
    if _sql_agent_instance is None:
        logger.info("Lazily initializing SQLAgent for tools.py...")
        _sql_agent_instance = SQLAgent()
    return _sql_agent_instance


def _get_web_agent() -> WebAgent:
    """Return the shared WebAgent instance, creating it on first use."""
    global _web_agent_instance
    if _web_agent_instance is None:
        logger.info("Lazily initializing WebAgent for tools.py...")
        _web_agent_instance = WebAgent()
    return _web_agent_instance


def peek_sql_agent() -> Optional[SQLAgent]:
    """
    Return the shared SQLAgent instance if it has already been created,
    without creating one.

    Unlike `_get_sql_agent()`, this never triggers initialization. It's
    used by callers (e.g. the orchestrator in agent.py) that want to pull
    the *structured* result (`SQLAgent.last_result`) left behind by a
    tool call that already ran within the same request — not to trigger
    a brand-new agent.
    """
    return _sql_agent_instance


def peek_web_agent() -> Optional[WebAgent]:
    """
    Return the shared WebAgent instance if it has already been created,
    without creating one. See `peek_sql_agent()` for why this exists.
    """
    return _web_agent_instance


# --------------------------------------------------------------------------- #
# Individual tool builders
# --------------------------------------------------------------------------- #

SQL_TOOL_NAME = "sql_database_tool"
SQL_TOOL_DESCRIPTION = (
    "Use this tool to answer questions about the company's internal business "
    "data stored in the SQL database — employees, departments, products, "
    "customers, and orders. Good for questions like 'show all employees', "
    "'what is the average salary', 'how many employees are there', "
    "'what is our total revenue', or 'show product sales'. "
    "Input should be a plain-language question; do not write raw SQL."
)


def build_sql_tool() -> Tool:
    """
    Build the SQL Tool: wraps SQLAgent.ask() so a LangChain agent can
    query the internal Employees/Departments/Products/Customers/Orders
    database using natural language.

    Returns:
        A LangChain `Tool` named "sql_database_tool".
    """

    def _run_sql_query(question: str) -> str:
        return _get_sql_agent().ask(question)

    return Tool.from_function(
        func=_run_sql_query,
        name=SQL_TOOL_NAME,
        description=SQL_TOOL_DESCRIPTION,
    )


WEB_SEARCH_TOOL_NAME = "web_search_tool"
WEB_SEARCH_TOOL_DESCRIPTION = (
    "Use this tool to answer questions that require current, real-world "
    "information from the internet — news, market trends, competitor "
    "information, or anything not stored in the internal database. "
    "Input should be a plain-language question or search topic. "
    "Returns a concise, summarized answer, not raw search results."
)


def build_web_search_tool() -> Tool:
    """
    Build the Web Search Tool: wraps WebAgent.ask() so a LangChain agent
    can research topics on the open web via Tavily Search + Groq
    summarization, with follow-up context preserved across calls.

    Returns:
        A LangChain `Tool` named "web_search_tool".
    """

    def _run_web_search(question: str) -> str:
        return _get_web_agent().ask(question)

    return Tool.from_function(
        func=_run_web_search,
        name=WEB_SEARCH_TOOL_NAME,
        description=WEB_SEARCH_TOOL_DESCRIPTION,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Central list of tool builder functions. To add a new tool to the agent
# in the future, write a new `build_*_tool()` function above and append
# it here — no other file needs to change.

_TOOL_BUILDERS = [
    build_sql_tool,
    build_web_search_tool,
]


def get_all_tools() -> List[Tool]:
    """
    Build and return the complete set of tools available to the
    orchestrator agent (agent.py).

    Returns:
        A list of LangChain `Tool` instances: [SQL Tool, Web Search Tool].
    """
    return [builder() for builder in _TOOL_BUILDERS]


# --------------------------------------------------------------------------- #
# Standalone execution (manual testing)
# --------------------------------------------------------------------------- #

def _run_demo() -> None:
    """Print the registered tools and run one sample call through each."""
    tools = get_all_tools()

    print("Registered tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description[:80]}...")

    sql_tool = next(t for t in tools if t.name == SQL_TOOL_NAME)
    web_tool = next(t for t in tools if t.name == WEB_SEARCH_TOOL_NAME)

    print("\n=== SQL Tool sample ===")
    print(sql_tool.func("How many employees are there?"))

    print("\n=== Web Search Tool sample ===")
    print(web_tool.func("What are the latest trends in the AI chip market?"))


if __name__ == "__main__":
    _run_demo()
