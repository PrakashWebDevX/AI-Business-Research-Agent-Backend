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
    - Tools are built using the `@tool` decorator, which derives a
      clean, correctly-named argument schema directly from each
      function's signature and docstring. This avoids the ambiguous
      `__arg1` fallback schema that the older `Tool.from_function(...)`
      pattern can produce.
    - The underlying agents (SQLAgent, WebAgent) are created lazily, on
      first use, not at import time. This means importing tools.py never
      fails just because an API key isn't set yet; the error only
      surfaces if that specific tool is actually invoked.
    - `get_all_tools()` is the single function the orchestrator should
      call to get the full toolset. Adding a new tool later only means
      writing a new `@tool`-decorated function and adding it to that list.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import Tool, tool

from sql_agent import SQLAgent
from web_agent import WebAgent

from rag import search_documents

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


DOCUMENT_SEARCH_TOOL_NAME = "document_search_tool"


@tool
def document_search_tool(question: str) -> str:
    """Use this tool for ANY question that should be answered using the
    user's uploaded documents (reports, PDFs, notes) rather than the
    internal database or the live web.

    Examples:
    - "What does the uploaded report say about..."
    - "Summarize the document I uploaded"
    - "According to my notes, what is..."
    - Any question referencing 'the document', 'the PDF', 'the report', 'my file'

    Always use this tool when the question references previously uploaded
    documents. Never answer these questions from your own knowledge or by
    guessing at document contents.
    """
    results = search_documents(question, top_k=4)
    if not results:
        return "No relevant information was found in the uploaded documents."

    parts = [f"[From '{r.filename}', chunk {r.chunk_index}]\n{r.content}" for r in results]
    return "\n\n".join(parts)


def _get_sql_agent() -> SQLAgent:
    """Return the shared SQLAgent instance, creating it on first use."""
    global _sql_agent_instance
    if _sql_agent_instance is None:
        logger.info("Lazily initializing SQLAgent for tools.py...")
        _sql_agent_instance = SQLAgent()
    return _sql_agent_instance


def _get_web_agent() -> WebAgent:
    global _web_agent_instance

    if _web_agent_instance is None:
        logger.info("Creating WebAgent...")
        _web_agent_instance = WebAgent()
        logger.info("WebAgent created successfully")

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
# Tool name constants
# --------------------------------------------------------------------------- #
# Kept as constants (rather than reading tool.name directly everywhere)
# because agent.py imports these to detect which tool(s) the orchestrator
# used for a given question. The @tool decorator below sets each
# function's `.name` to match these automatically.

SQL_TOOL_NAME = "sql_database_tool"
WEB_SEARCH_TOOL_NAME = "web_search_tool"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

@tool
def sql_database_tool(question: str) -> str:
    """Use this tool for ANY question about the company's internal database.

    Examples:
    - employees
    - salary
    - revenue
    - products
    - orders
    - customers
    - departments
    - sales
    - count
    - average
    - total

    Always use this tool whenever the answer may exist in the SQL database.
    Never answer these questions from your own knowledge.
    """
    return _get_sql_agent().ask(question)


@tool
def web_search_tool(question: str) -> str:
    """Use this tool for ANY question requiring internet knowledge.

    Examples:
    - latest news
    - market trends
    - competitors
    - startup funding
    - company information
    - stock market
    - AI news
    - industry reports
    - public companies
    - current events

    Always use this tool whenever current or public information is required.
    Never answer these questions using your own knowledge.
    """
    logger.info("WEB TOOL CALLED")
    logger.info("Question: %s", question)
    result = _get_web_agent().ask(question)
    logger.info("WEB TOOL RESULT: %s", result)
    return result


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Central list of tools. To add a new tool to the agent in the future,
# write a new `@tool`-decorated function above and append it here — no
# other file needs to change.

def get_all_tools() -> List[Tool]:
    """
    Return the complete set of tools available to the orchestrator agent
    (agent.py).

    Returns:
        A list of LangChain `Tool` instances: [SQL Tool, Web Search Tool].
    """
    return [sql_database_tool, web_search_tool, document_search_tool]


# --------------------------------------------------------------------------- #
# Standalone execution (manual testing)
# --------------------------------------------------------------------------- #

def _run_demo() -> None:
    """Print the registered tools and run one sample call through each."""
    tools = get_all_tools()

    print("Registered tools:")
    for t in tools:
        print(f"- {t.name}: {t.description[:80]}...")

    sql_tool = next(t for t in tools if t.name == SQL_TOOL_NAME)
    web_tool = next(t for t in tools if t.name == WEB_SEARCH_TOOL_NAME)

    print("\n=== SQL Tool sample ===")
    print(sql_tool.invoke("How many employees are there?"))

    print("\n=== Web Search Tool sample ===")
    print(web_tool.invoke("What are the latest trends in the AI chip market?"))


if __name__ == "__main__":
    _run_demo()