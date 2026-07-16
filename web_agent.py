"""
web_agent.py

Interface-adapter layer for the AI Business Research Agent.

Responsibilities:
    - Wrap the Tavily Search API as a LangChain `Tool` so an LLM agent can
      call it to search the internet for current, real-world information
      (news, company facts, market data, anything outside the local
      SQLite database).
    - Drive that tool with Groq (`ChatGroq`) via a
      LangChain tool-calling agent.
    - Maintain short-term conversational memory so the agent can answer
      natural follow-up questions ("What about last year?", "And their
      main competitor?") without the caller having to re-supply context.
    - Always return concise, summarized answers rather than raw search
      results or long-winded prose.

This module is designed to be consumed by agent.py (the orchestrator),
but can also be run standalone for manual testing:

    python web_agent.py
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Final, List, Optional

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import Tool, tool
from langchain_groq import ChatGroq
from tavily import TavilyClient

from prompts import get_web_agent_prompt
from schemas import ToolExecutionResult
from utils import extract_output_text

# --------------------------------------------------------------------------- #
# Environment & Logging
# --------------------------------------------------------------------------- #

load_dotenv()  # Load GROQ_API_KEY and TAVILY_API_KEY from .env into os.environ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

# NOTE: These will eventually be centralized in config.py. They are kept
# local for now so this module stays fully functional on its own.
GROQ_MODEL_NAME: Final[str] = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_TEMPERATURE: Final[float] = 0.2  # Slight creativity for natural summaries, still mostly factual.
WEB_AGENT_MAX_ITERATIONS: Final[int] = 8
TAVILY_MAX_RESULTS: Final[int] = 5
MAX_HISTORY_MESSAGES: Final[int] = 12  # Cap on stored chat turns to keep context small and fast.

# NOTE: The system prompt that steers this agent toward searching,
# summarizing, and staying concise now lives in prompts.py as
# WEB_AGENT_SYSTEM_PROMPT (used inside get_web_agent_prompt()), so it can
# be reviewed and tuned alongside every other prompt in the project.


# --------------------------------------------------------------------------- #
# Tavily search tool (wrapped as a LangChain Tool)
# --------------------------------------------------------------------------- #
# NOTE: This is built as an instance method (WebAgent._build_tavily_tool)
# rather than a standalone function, so the search closure can record
# each result's title/url onto `self._last_sources` — giving
# `ask_structured()` a "Sources" list to return to the caller (e.g. the
# dashboard UI), without changing what the LLM itself receives back from
# the tool call.
#
# The inner `_search` function is wrapped with the `@tool` decorator
# (rather than `Tool.from_function`), which derives a clean, correctly
# named `query` parameter schema directly from the function signature.
# `Tool.from_function` without an explicit args_schema falls back to an
# ambiguous single-argument schema that Groq's tool-calling can reject
# with a "missing properties: '__arg1'" validation error — @tool avoids
# that entirely.


# --------------------------------------------------------------------------- #
# Web Agent wrapper
# --------------------------------------------------------------------------- #

class WebAgent:
    """
    Encapsulates a LangChain tool-calling agent that searches the web via
    Tavily and answers using Groq, with short-term memory so it
    can handle natural follow-up questions.

    Usage:
        web_agent = WebAgent()
        answer = web_agent.ask("What is Nvidia's latest quarterly revenue?")
        follow_up = web_agent.ask("How does that compare to last quarter?")
    """

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        temperature: float = GROQ_TEMPERATURE,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the web research agent.

        Args:
            model_name: Groq model identifier (e.g. "openai/gpt-oss-120b").
            temperature: Sampling temperature for the LLM.
            verbose: If True, prints the agent's intermediate reasoning
                and tool calls (useful for debugging).

        Raises:
            EnvironmentError: If GROQ_API_KEY or TAVILY_API_KEY is missing.
        """
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before creating a WebAgent."
            )

        # 1. Tool: Tavily search, wrapped as a LangChain Tool. Built as an
        #    instance method so its search closure can record sources
        #    onto self._last_sources (see _build_tavily_tool below).
        self._last_sources: List[Dict[str, str]] = []
        self._search_tool = self._build_tavily_tool()

        # 2. LLM: Groq via LangChain's chat model wrapper.
        self._llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=1,      # fail fast instead of waiting ~56s per retry
            timeout=20,          # give up after 20s instead of hanging indefinitely
        )

        # 3. Prompt: system instructions + a chat history placeholder (for
        #    follow-up questions) + the current input + the agent's
        #    scratchpad (required by tool-calling agents to track its
        #    own intermediate tool calls within a single turn). Built
        #    centrally in prompts.py.
        self._prompt = get_web_agent_prompt()

        # 4. Agent + Executor: binds the LLM, tool, and prompt together
        #    into a runnable that can decide when to search vs. answer
        #    directly from the conversation.
        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=[self._search_tool],
            prompt=self._prompt,
        )
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=[self._search_tool],
            verbose=verbose,
            max_iterations=WEB_AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
        )

        # In-memory chat history, used to support follow-up questions
        # across successive calls to `ask()`. Kept as a simple list of
        # LangChain message objects rather than a full memory class, to
        # keep this module dependency-light and easy to reason about.
        self._chat_history: List[BaseMessage] = []

        # The most recent structured result, populated by ask_structured().
        # Exposed so callers like tools.py / agent.py can pull rich
        # metadata (sources, timing) after a tool call that only returns
        # plain text to the orchestrating LLM.
        self.last_result: Optional[ToolExecutionResult] = None

        logger.info("WebAgent initialized with model '%s'.", model_name)

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def ask_structured(self, question: str) -> ToolExecutionResult:
        """
        Ask the web agent a question and return a full structured result:
        the answer text, the web sources used, and execution time.

        Args:
            question: A natural-language question, e.g. "What's the
                latest news on the semiconductor industry?" or, as a
                follow-up, "How might that affect smartphone prices?".

        Returns:
            A `ToolExecutionResult`. On failure, `answer` contains a
            friendly error message and `sources` is left empty, so
            callers never need to handle an exception directly.
        """
        logger.info("WebAgent received question: %s", question)
        start_time = time.perf_counter()
        self._last_sources = []  # Reset before this turn's search(es) run.

        try:
            result = self._agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": self._chat_history,
                }
            )
            answer = extract_output_text(result.get("output", "")).strip()
            answer = answer or "I couldn't find a clear answer to that from the web."

            # Update memory with this turn, then trim to keep it bounded.
            self._chat_history.append(HumanMessage(content=question))
            self._chat_history.append(AIMessage(content=answer))
            self._chat_history = self._chat_history[-MAX_HISTORY_MESSAGES:]

            structured = ToolExecutionResult(
                answer=answer,
                TOOL_LABEL_WEB: Final[str] = "Web Research",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
                sources=list(self._last_sources) or None,
            )
        except OutputParserException as exc:
            logger.exception("Failed to parse agent output: %s", exc)
            structured = ToolExecutionResult(
                answer="I had trouble interpreting the search results. Please try rephrasing your question.",
                TOOL_LABEL_WEB: Final[str] = "Web Research",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure as a safe message
            logger.exception("WebAgent failed to answer question: %s", question)
            structured = ToolExecutionResult(
                answer=f"Sorry, I ran into an error while searching the web: {exc}",
                TOOL_LABEL_WEB: Final[str] = "Web Research",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )

        self.last_result = structured
        return structured

    def ask(self, question: str) -> str:
        """
        Ask the web agent a question. The agent will search the internet
        if needed, summarize what it finds, and return a concise answer.
        Prior questions and answers in this session are used as context,
        so natural follow-up questions are supported.

        This is a thin convenience wrapper around `ask_structured()`. It
        also has the side effect of updating `self.last_result`, so a
        caller that needs richer metadata afterward (e.g. the FastAPI
        backend) can read `web_agent.last_result`.

        Args:
            question: A natural-language question.

        Returns:
            A concise, plain-language answer string.
        """
        return self.ask_structured(question).answer

    def reset_memory(self) -> None:
        """Clear the conversation history, starting a fresh research session."""
        self._chat_history.clear()
        logger.info("WebAgent conversation memory cleared.")

    # ----------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------- #

    def _build_tavily_tool(self) -> Tool:
        """
        Build a LangChain `Tool` that wraps the Tavily Search API.

        A hand-written function is used (rather than the prebuilt
        community tool class) so we have full control over the
        input/output formatting fed back to the LLM, and so we can
        record each result's title/url onto `self._last_sources` for
        `ask_structured()` to expose afterward. The `@tool` decorator
        (rather than `Tool.from_function`) ensures Groq receives a
        clean, unambiguous `query` parameter schema.

        Returns:
            A `Tool` instance named "tavily_web_search" ready to be
            handed to a LangChain agent.

        Raises:
            EnvironmentError: If TAVILY_API_KEY is not set.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before creating a WebAgent."
            )

        client = TavilyClient(api_key=api_key)

        @tool(
            "tavily_web_search",
            description=(
                "Search the internet for current, real-world information such as "
                "news, company facts, market trends, or anything not available in "
                "the local database. Input should be a focused search query string."
            ),
        )
        def _search(query: str) -> str:
            logger.info("=" * 60)
            logger.info("TAVILY SEARCH QUERY: %s", query)
            try:
                response = client.search(
                    query=query,
                    max_results=TAVILY_MAX_RESULTS,
                    include_answer=True,
                )
            except Exception as exc:  # noqa: BLE001 - surface search failures as tool output, not a crash
                logger.exception("Tavily search failed for query: %s", query)
                return f"Web search failed due to an error: {exc}"

            logger.info("TAVILY RESPONSE:")
            logger.info(response)
            logger.info("=" * 60)

            parts = []
            quick_answer = response.get("answer")
            if quick_answer:
                parts.append(f"Quick answer: {quick_answer}")

            for i, result in enumerate(response.get("results", []), start=1):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")[:500]  # cap each result to ~500 chars before feeding to the LLM
                logger.info("RESULT %s", i)
                logger.info("Title: %s", title)
                logger.info("URL: %s", url)
                parts.append(f"[{i}] {title} ({url})\n{content}")
                self._last_sources.append({
                    "title": title,
                    "url": url,
                })

            return "\n\n".join(parts) if parts else "No relevant results were found."

        return _search


# --------------------------------------------------------------------------- #
# Standalone execution (manual testing / demo)
# --------------------------------------------------------------------------- #

def _run_demo() -> None:
    """Run a short multi-turn conversation to demonstrate search + follow-ups."""
    agent = WebAgent(verbose=True)

    print("\n=== Initial question ===")
    print(agent.ask("What are the latest trends in the AI chip market?"))

    print("\n=== Follow-up question ===")
    print(agent.ask("Which company is currently leading in that space?"))


if __name__ == "__main__":
    _run_demo()