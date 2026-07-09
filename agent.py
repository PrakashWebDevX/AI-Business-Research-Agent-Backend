"""
agent.py

Application / orchestration layer for the AI Business Research Agent.

Responsibilities:
    - Act as the single "brain" of the system: given a user's question,
      automatically decide — with no manual routing and no user prompt
      for tool choice — whether to consult the internal SQL database,
      search the web, use both, or answer directly.
    - Wire together the LLM (Groq), the registered tools (SQL Tool and
      Web Search Tool from tools.py), and the orchestrator system prompt
      (from prompts.py) into a single LangChain `AgentExecutor` built via
      tool calling.
    - Maintain conversational memory so multi-turn, mixed-topic
      conversations (some questions internal, some external) work
      naturally.
    - Return clean, final answers only — never raw tool output, raw SQL,
      or the model's internal tool-selection reasoning.

This module depends only on tools.py and prompts.py, never directly on
sql_agent.py or web_agent.py — the orchestrator doesn't need to know how
each tool is implemented, only that it exists and what it's for. This
keeps agent.py decoupled and easy to extend with new tools later.

Usage (from another module, e.g. app.py):

    from agent import BusinessAgent

    business_agent = BusinessAgent()
    answer = business_agent.ask("What is our average salary?")

Can also be run standalone for manual testing:

    python agent.py
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
from langchain_groq import ChatGroq

from prompts import get_orchestrator_prompt
from schemas import ToolExecutionResult
from tools import SQL_TOOL_NAME, WEB_SEARCH_TOOL_NAME, get_all_tools, peek_sql_agent, peek_web_agent
from utils import extract_output_text

# --------------------------------------------------------------------------- #
# Environment & Logging
# --------------------------------------------------------------------------- #

load_dotenv()  # Load GROQ_API_KEY / TAVILY_API_KEY from .env into os.environ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

# NOTE: These will eventually be centralized in config.py. They are kept
# local for now so this module stays fully functional on its own.
GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME",
    "llama-3.3-70b-versatile"
)
GROQ_TEMPERATURE: Final[float] = 0.1  # Low temperature: consistent tool-selection decisions.
ORCHESTRATOR_MAX_ITERATIONS: Final[int] = 10
MAX_HISTORY_MESSAGES: Final[int] = 12  # Cap on stored chat turns to keep context small and fast.


# --------------------------------------------------------------------------- #
# Business Agent (Orchestrator)
# --------------------------------------------------------------------------- #

class BusinessAgent:
    """
    The top-level orchestrator agent for the AI Business Research Agent.

    Wraps a LangChain tool-calling `AgentExecutor` that has access to two
    tools — the SQL Tool and the Web Search Tool (see tools.py) — and
    autonomously decides which one(s) to invoke for any given question,
    based purely on the system prompt's instructions. No routing logic
    lives in this file: tool selection is entirely delegated to the LLM.

    Usage:
        agent = BusinessAgent()
        answer = agent.ask("Show all employees.")
        follow_up = agent.ask("How does that compare to industry averages?")
    """

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        temperature: float = GROQ_TEMPERATURE,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the orchestrator agent.

        Args:
            model_name: Groq model identifier (e.g. "openai/gpt-oss-120b").
            temperature: Sampling temperature for the orchestrator LLM.
                Kept low so tool-selection behavior is consistent.
            verbose: If True, prints the agent's intermediate reasoning
                and tool calls (useful for debugging which tool was
                chosen and why).

        Raises:
            EnvironmentError: If GROQ_API_KEY is not set.
        """
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before creating a BusinessAgent."
            )

        # 1. LLM: Groq drives the orchestrator's reasoning and
        #    tool-selection decisions.
        self._llm = ChatGroq(
            model=model_name,
            temperature=temperature,
        )

        # 2. Tools: pulled from the central registry in tools.py. Adding
        #    a new capability to the whole system later requires no
        #    changes here — just registering it in tools.py.
        self._tools = get_all_tools()
        logger.info("=" * 60)
        logger.info("REGISTERED TOOLS")

        for tool in self._tools:
            logger.info("Tool: %s", tool.name)
            logger.info("Description: %s", tool.description)

        logger.info("=" * 60)

        # 3. Prompt: the orchestrator system prompt from prompts.py,
        #    which instructs the model to autonomously choose between
        #    the SQL Tool and the Web Search Tool with no manual routing
        #    and no asking the user to choose.
        self._prompt = get_orchestrator_prompt()

        # 4. Agent + Executor: LangChain's tool-calling agent binds the
        #    LLM to the tools via native function/tool calling (as
        #    opposed to older ReAct-style text parsing), and the
        #    AgentExecutor drives the actual reasoning -> tool call ->
        #    observation -> final answer loop.
        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=self._prompt,
        )
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=verbose,
            max_iterations=ORCHESTRATOR_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True,  # Needed to detect which tool(s) ran (see ask_structured()).
        )

        # Conversation memory, so follow-up questions (whether SQL-based,
        # web-based, or a mix) retain context across calls to ask().
        self._chat_history: List[BaseMessage] = []

        # The most recent structured result, populated by ask_structured().
        self.last_result: Optional[ToolExecutionResult] = None

        logger.info(
            "BusinessAgent initialized with model '%s' and tools: %s",
            model_name,
            [tool.name for tool in self._tools],
        )

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def ask_structured(self, question: str) -> ToolExecutionResult:
        """
        Ask the orchestrator a business question and return a full
        structured result: the clean final answer, which tool(s) were
        used, total execution time, and — merged in from whichever
        sub-agent(s) ran — the generated SQL, table data, chart, and/or
        web sources.

        Args:
            question: A natural-language business question.

        Returns:
            A `ToolExecutionResult`. On failure, `answer` contains a
            friendly error message and the other fields are left empty.
        """
        logger.info("BusinessAgent received question: %s", question)
        start_time = time.perf_counter()
        try:
            result = self._agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": self._chat_history,
                }
            )
            logger.info("=" * 60)
            logger.info("Agent Result:")
            logger.info(result)
            logger.info("Intermediate Steps:")
            logger.info(result.get("intermediate_steps"))
            logger.info("=" * 60)
            answer = self._clean_response(result.get("output", ""))
            # Update memory with this turn, then trim to keep it bounded.
            self._chat_history.append(HumanMessage(content=question))
            self._chat_history.append(AIMessage(content=answer))
            self._chat_history = self._chat_history[-MAX_HISTORY_MESSAGES:]
            structured = self._build_structured_result(
                answer=answer,
                intermediate_steps=result.get("intermediate_steps", []),
                elapsed_seconds=round(time.perf_counter() - start_time, 2),
            )
        except OutputParserException as exc:
            logger.error("Failed to parse agent output: %s", exc)
            structured = ToolExecutionResult(
                answer="I had trouble interpreting the results. Please try rephrasing your question.",
                tool_used="Direct Answer",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure as a safe message
            logger.exception("BusinessAgent failed to answer question: %s", question)
            structured = ToolExecutionResult(
                answer=f"Sorry, I ran into an error while processing your request: {exc}",
                tool_used="Direct Answer",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )

        self.last_result = structured
        return structured

    def ask(self, question: str) -> str:
        """
        Ask the orchestrator a business question. The agent automatically
        decides — with no manual routing — whether to query the internal
        SQL database, search the web, use both, or answer directly, then
        returns a single, clean final answer.

        This is a thin convenience wrapper around `ask_structured()` for
        callers (e.g. the CLI in app.py) that only need the text answer.
        Callers that need richer metadata (e.g. the FastAPI backend for
        the dashboard) should call `ask_structured()` directly instead.

        Args:
            question: A natural-language business question, e.g.
                "What is our average salary?" or "What's the latest news
                on our biggest competitor?" or a mix of both.

        Returns:
            A clean, final answer string. Raw SQL, raw search results,
            and the model's tool-selection reasoning are never included.
            On failure, a friendly error message is returned instead of
            raising, so callers (e.g. app.py) can display something
            sensible to the end user.
        """
        return self.ask_structured(question).answer

    def reset_memory(self) -> None:
        """Clear the conversation history, starting a fresh session."""
        self._chat_history.clear()
        logger.info("BusinessAgent conversation memory cleared.")

    def load_history(self, turns: List[Dict[str, str]]) -> None:
        """
        Replace the agent's conversation memory with a specific sequence
        of prior turns.

        This exists for callers that manage multiple independent
        conversations against a single shared `BusinessAgent` instance —
        most notably the FastAPI dashboard backend, which lets a user
        switch between saved chat sessions. Before continuing a
        previously-started session, the backend calls this method to
        restore that session's history, so follow-up questions resolve
        against the right context instead of whatever conversation was
        last active.

        Args:
            turns: A list of {"role": "user" | "assistant", "content": str}
                dicts, oldest first. Only the most recent
                `MAX_HISTORY_MESSAGES` are kept.
        """
        messages: List[BaseMessage] = []
        for turn in turns:
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=turn.get("content", "")))
            else:
                messages.append(AIMessage(content=turn.get("content", "")))
        self._chat_history = messages[-MAX_HISTORY_MESSAGES:]

    # ----------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------- #

    @staticmethod
    def _build_structured_result(
        answer: str,
        intermediate_steps: List[Any],
        elapsed_seconds: float,
    ) -> ToolExecutionResult:
        """
        Determine which tool(s) the orchestrator used this turn, and
        merge in any structured metadata (generated SQL, table data,
        chart, web sources) that the corresponding sub-agent(s) left
        behind in their own `last_result`.

        Args:
            answer: The orchestrator's own clean, final answer text.
            intermediate_steps: The `intermediate_steps` list from the
                AgentExecutor result — used only to detect *which* tools
                ran, never to extract raw text shown to the user.
            elapsed_seconds: Total wall-clock time for this turn.

        Returns:
            A `ToolExecutionResult` with `tool_used` set to "SQL Agent",
            "Web Research", "Mixed", or "Direct Answer", and any relevant
            sub-agent metadata merged in.
        """
        tools_called = {getattr(action, "tool", None) for action, _obs in intermediate_steps}
        used_sql = SQL_TOOL_NAME in tools_called
        used_web = WEB_SEARCH_TOOL_NAME in tools_called

        if used_sql and used_web:
            tool_used = "Mixed"
        elif used_sql:
            tool_used = "SQL Agent"
        elif used_web:
            tool_used = "Web Research"
        else:
            tool_used = "Direct Answer"

        generated_sql = None
        table_data = None
        chart = None
        sources = None

        if used_sql:
            sql_agent = peek_sql_agent()
            sql_result = sql_agent.last_result if sql_agent else None
            if sql_result:
                generated_sql = sql_result.generated_sql
                table_data = sql_result.table_data
                chart = sql_result.chart

        if used_web:
            web_agent = peek_web_agent()
            web_result = web_agent.last_result if web_agent else None
            if web_result:
                sources = web_result.sources

        return ToolExecutionResult(
            answer=answer,
            tool_used=tool_used,
            execution_time_seconds=elapsed_seconds,
            generated_sql=generated_sql,
            table_data=table_data,
            sources=sources,
            chart=chart,
        )

    # ----------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------- #

    @staticmethod
    def _clean_response(raw_output: Any) -> str:
        """
        Normalize the agent's final output into a clean, presentable string.

        Args:
            raw_output: The raw `output` field from the AgentExecutor
                result. Depending on the LLM provider and whether
                tool-calling/streaming was used, this can be a plain
                string or a list of content blocks (some providers,
                depending on streaming/tool-calling mode, return
                `[{"type": "text", "text": "..."}]` instead of a
                string). `extract_output_text()` handles both shapes.

        Returns:
            A trimmed string, with a friendly fallback if the agent
            produced no usable content.
        """
        cleaned = extract_output_text(raw_output).strip()
        return cleaned or "I couldn't find a clear answer to that question."


# --------------------------------------------------------------------------- #
# Standalone execution (manual testing / demo)
# --------------------------------------------------------------------------- #

def _run_demo() -> None:
    """
    Demonstrate automatic tool selection across SQL-only, web-only, and
    mixed questions, with no manual routing performed by this script.
    """
    agent = BusinessAgent(verbose=True)

    demo_questions = [
        "How many employees do we have, and what is the average salary?",
        "What are the latest trends in the AI chip market?",
        "What is our total revenue, and how does that compare to industry growth trends this year?",
    ]

    for question in demo_questions:
        print(f"\n=== Question: {question} ===")
        print(agent.ask(question))


if __name__ == "__main__":
    _run_demo()
