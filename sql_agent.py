"""
sql_agent.py

Interface-adapter layer for the AI Business Research Agent.

Responsibilities:
    - Wrap the SQLite database (database/employee.db) in a LangChain
      `SQLDatabase` object.
    - Expose that database to an LLM (Groq, via
      `ChatGroq`) through LangChain's `SQLDatabaseToolkit`,
      which auto-generates tools such as "list tables", "describe schema",
      "run query", and "check query" for the agent to call.
    - Build a ready-to-use SQL agent (`AgentExecutor`) that can answer
      natural-language business questions by writing and executing SQL
      against the Employees / Departments / Products / Customers / Orders
      tables created in database.py.

This module is designed to be consumed by agent.py (the orchestrator),
but can also be run standalone for testing:

    python sql_agent.py

Supported example questions (see README for more):
    - "Show all employees."
    - "What is the top salary?"
    - "What is the average salary?"
    - "How many employees are there?"
    - "What is the total revenue?"
    - "Show product sales."
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Final, List, Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from sqlalchemy import text

from database import get_engine, init_db
from prompts import SQL_AGENT_SYSTEM_PROMPT
from schemas import ChartSpec, ToolExecutionResult
from utils import extract_output_text

# --------------------------------------------------------------------------- #
# Environment & Logging
# --------------------------------------------------------------------------- #

load_dotenv()  # Load GROQ_API_KEY (and other secrets) from .env into os.environ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

# NOTE: These will eventually be centralized in config.py. They are kept
# local for now so this module stays fully functional on its own.
GROQ_MODEL_NAME: Final[str] = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_TEMPERATURE: Final[float] = 0.0  # Deterministic output — this is a data agent, not a creative one.
SQL_AGENT_MAX_ITERATIONS: Final[int] = 15
SQL_AGENT_TOP_K: Final[int] = 25  # Max rows the agent will consider per query result.

# NOTE: The system-level prefix that steers the agent's SQL behavior
# (read-only constraints, revenue/sales definitions, etc.) now lives in
# prompts.py as SQL_AGENT_SYSTEM_PROMPT, so it can be reviewed and tuned
# alongside every other prompt in the project.


# --------------------------------------------------------------------------- #
# SQL Agent wrapper
# --------------------------------------------------------------------------- #

class SQLAgent:
    """
    Encapsulates a LangChain SQL agent backed by ChatGroq
    and SQLDatabaseToolkit.

    Usage:
        sql_agent = SQLAgent()
        answer = sql_agent.ask("Show all employees.")
    """

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        temperature: float = GROQ_TEMPERATURE,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SQL agent.

        Args:
            model_name: Groq model identifier (e.g. "openai/gpt-oss-120b").
            temperature: Sampling temperature for the LLM. Kept at 0.0 by
                default for consistent, reproducible SQL generation.
            verbose: If True, prints the agent's intermediate reasoning
                and tool calls (useful for debugging).

        Raises:
            EnvironmentError: If GROQ_API_KEY is not set.
        """
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before creating a SQLAgent."
            )

        # Ensure the database exists and is seeded before the agent tries
        # to query it. init_db() is idempotent, so this is safe to call
        # every time the agent starts up.
        init_db()

        # 1. LLM: Groq via LangChain's chat model wrapper.
        self._llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=1,      # fail fast instead of waiting ~56s per retry
            timeout=20,          # give up after 20s instead of hanging indefinitely
        )

        # 2. Database: wrap the existing SQLAlchemy engine from database.py
        #    so both modules share a single source of truth for the schema
        #    and connection pooling.
        self._db = SQLDatabase(engine=get_engine())

        # 3. Toolkit: auto-generates the standard SQL tools (list tables,
        #    get schema, query, query-checker) bound to our LLM + DB.
        self._toolkit = SQLDatabaseToolkit(db=self._db, llm=self._llm)

        # 4. Agent: a ready-to-run AgentExecutor that can reason over the
        #    toolkit's tools to answer natural-language questions.
        #    `return_intermediate_steps=True` lets us inspect exactly which
        #    SQL query the agent ran (for display/export in the dashboard).
        self._agent_executor = create_sql_agent(
            llm=self._llm,
            toolkit=self._toolkit,
            agent_type="tool-calling",
            prefix=SQL_AGENT_SYSTEM_PROMPT,
            top_k=SQL_AGENT_TOP_K,
            max_iterations=SQL_AGENT_MAX_ITERATIONS,
            verbose=verbose,
            agent_executor_kwargs={"return_intermediate_steps": True},
        )

        # The most recent structured result, populated by ask_structured().
        # Exposed so callers like tools.py / agent.py can pull rich
        # metadata (generated SQL, table rows, chart) after a tool call
        # that only returns plain text to the orchestrating LLM.
        self.last_result: Optional[ToolExecutionResult] = None

        logger.info("SQLAgent initialized with model '%s'.", model_name)

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def ask_structured(self, question: str) -> ToolExecutionResult:
        """
        Ask the SQL agent a natural-language business question and return
        a full structured result: the answer text, the generated SQL (if
        any), the raw result rows (for tables/exports), an auto-suggested
        chart (if the data is chart-friendly), and execution time.

        Args:
            question: A natural-language question, e.g. "What is the
                average salary?" or "Show product sales."

        Returns:
            A `ToolExecutionResult`. On failure, `answer` contains a
            friendly error message and the other fields are left empty,
            so callers never need to handle an exception directly.
        """
        logger.info("SQLAgent received question: %s", question)
        start_time = time.perf_counter()

        try:
            result = self._agent_executor.invoke({"input": question})
            answer = extract_output_text(result.get("output", "")).strip()
            answer = answer or "I couldn't find an answer to that question in the database."

            generated_sql = self._extract_generated_sql(result.get("intermediate_steps", []))
            table_data = self._fetch_rows_for_sql(generated_sql) if generated_sql else None
            chart = self._maybe_build_chart(table_data) if table_data else None

            structured = ToolExecutionResult(
                answer=answer,
                tool_used="SQL Agent",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
                generated_sql=generated_sql,
                table_data=table_data,
                chart=chart,
            )
        except OutputParserException as exc:
            logger.error("Failed to parse agent output: %s", exc)
            structured = ToolExecutionResult(
                answer="I had trouble interpreting the database results. Please try rephrasing your question.",
                tool_used="SQL Agent",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure as a safe message
            logger.exception("SQLAgent failed to answer question: %s", question)
            structured = ToolExecutionResult(
                answer=f"Sorry, I ran into an error while querying the database: {exc}",
                tool_used="SQL Agent",
                execution_time_seconds=round(time.perf_counter() - start_time, 2),
            )

        self.last_result = structured
        return structured

    def ask(self, question: str) -> str:
        """
        Ask the SQL agent a natural-language business question and get
        back a plain-language answer derived from the database.

        This is a thin convenience wrapper around `ask_structured()` for
        callers (e.g. the CLI, or the orchestrator's tool wiring) that
        only need the text answer. It also has the side effect of
        updating `self.last_result` with the full structured result, so
        a caller that needs richer metadata afterward (e.g. the FastAPI
        backend) can read `sql_agent.last_result`.

        Args:
            question: A natural-language question.

        Returns:
            The agent's final answer as a plain string.
        """
        return self.ask_structured(question).answer

    # ----------------------------------------------------------------- #
    # Internal helpers for structured result extraction
    # ----------------------------------------------------------------- #

    @staticmethod
    def _extract_generated_sql(intermediate_steps: List[Any]) -> Optional[str]:
        """
        Find the SQL query text the agent actually executed, by scanning
        the AgentExecutor's intermediate steps for a call to the
        `sql_db_query` tool (the one that runs SQL against the database).

        Args:
            intermediate_steps: The `intermediate_steps` list from an
                `AgentExecutor.invoke(...)` result — a list of
                (AgentAction, observation) tuples.

        Returns:
            The most recent SQL query string the agent ran, or None if
            no query tool was called (e.g. the agent answered from
            schema inspection alone, or failed before querying).
        """
        generated_sql: Optional[str] = None
        for action, _observation in intermediate_steps:
            tool_name = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", None)
            if tool_name == "sql_db_query" and tool_input:
                # tool_input may be a raw string or a dict like {"query": "..."}
                if isinstance(tool_input, dict):
                    generated_sql = tool_input.get("query") or tool_input.get("input")
                else:
                    generated_sql = str(tool_input)
        return generated_sql.strip() if generated_sql else None

    def _fetch_rows_for_sql(self, sql: str) -> Optional[List[Dict[str, Any]]]:
        """
        Re-execute a SELECT query directly via SQLAlchemy to get clean,
        structured rows (list of dicts) for table display and export.

        We re-run the query ourselves (rather than parsing the agent's
        stringified tool observation) because the observation is a
        Python-repr string of a list of tuples with no column names —
        not reliable enough for a data table or CSV/Excel export.

        Args:
            sql: The SQL string the agent executed.

        Returns:
            A list of {column_name: value} dicts, or None if the query
            isn't a safe read-only SELECT, or re-execution fails for any
            reason (in which case the text answer is still returned to
            the user; only the table/export data is skipped).
        """
        normalized = sql.strip().lower()
        if not normalized.startswith("select"):
            # Defense in depth: only ever re-run SELECT statements here,
            # even though the agent is already instructed to be read-only.
            logger.warning("Skipping row fetch for non-SELECT statement: %s", sql)
            return None

        try:
            with get_engine().connect() as connection:
                cursor_result = connection.execute(text(sql))
                columns = list(cursor_result.keys())
                rows = [dict(zip(columns, row)) for row in cursor_result.fetchall()]
                return rows
        except Exception:  # noqa: BLE001 - table data is a bonus, never fatal
            logger.exception("Failed to re-execute SQL for structured row data: %s", sql)
            return None

    @staticmethod
    def _maybe_build_chart(rows: List[Dict[str, Any]]) -> Optional[ChartSpec]:
        """
        Heuristically decide whether the query result is chart-friendly,
        and if so, build a simple bar chart from it.

        Current heuristic: exactly two columns, where the second column
        is entirely numeric and there are between 2 and 15 rows. This
        covers the most common business-question shapes (e.g. "sales by
        product", "headcount by department") without trying to be
        exhaustively clever.

        Args:
            rows: The structured query result rows.

        Returns:
            A `ChartSpec` (bar chart) if the shape fits, otherwise None.
        """
        if not rows or not (2 <= len(rows) <= 15):
            return None

        columns = list(rows[0].keys())
        if len(columns) != 2:
            return None

        label_col, value_col = columns
        values: List[float] = []
        for row in rows:
            value = row.get(value_col)
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    return None  # Second column isn't numeric — not chartable.

        labels = [str(row.get(label_col)) for row in rows]
        return ChartSpec(
            chart_type="bar",
            title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
            labels=labels,
            values=values,
        )

    # ----------------------------------------------------------------- #
    # Convenience methods for the required demo questions.
    # These simply route to `ask()` with well-formed prompts, so callers
    # (or the orchestrator agent) can invoke them directly without having
    # to remember exact phrasing.
    # ----------------------------------------------------------------- #

    def show_all_employees(self) -> str:
        """Return a summary of all employees in the database."""
        return self.ask("Show all employees, including their name, job title, and department.")

    def get_top_salary(self) -> str:
        """Return the highest salary paid, and who earns it."""
        return self.ask("What is the top (highest) salary, and which employee earns it?")

    def get_average_salary(self) -> str:
        """Return the average salary across all employees."""
        return self.ask("What is the average salary across all employees?")

    def get_employee_count(self) -> str:
        """Return the total number of employees."""
        return self.ask("How many employees are there in total?")

    def get_revenue(self) -> str:
        """Return total revenue generated from all orders."""
        return self.ask("What is the total revenue generated from all orders?")

    def get_product_sales(self) -> str:
        """Return a breakdown of sales performance per product."""
        return self.ask(
            "Show product sales: for each product, report total quantity sold "
            "and total revenue generated, ordered from highest to lowest revenue."
        )


# --------------------------------------------------------------------------- #
# Standalone execution (manual testing / demo)
# --------------------------------------------------------------------------- #

def _run_demo() -> None:
    """Run through the required set of example questions and print answers."""
    agent = SQLAgent(verbose=True)

    demo_questions = [
        ("Show all employees.", agent.show_all_employees),
        ("Top salary.", agent.get_top_salary),
        ("Average salary.", agent.get_average_salary),
        ("Employee count.", agent.get_employee_count),
        ("Revenue.", agent.get_revenue),
        ("Product sales.", agent.get_product_sales),
    ]

    for label, fn in demo_questions:
        print(f"\n=== {label} ===")
        print(fn())


if __name__ == "__main__":
    _run_demo()
