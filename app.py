"""
app.py

Presentation layer for the AI Business Research Agent.

Responsibilities:
    - Provide a simple command-line interface (CLI) for interacting with
      the BusinessAgent orchestrator (agent.py).
    - Read user questions, pass them to the agent, and print the
      returned answers.
    - Support multi-turn conversation: the underlying BusinessAgent
      retains chat history across turns, so follow-up questions work
      naturally within a single session.
    - Handle startup errors (e.g. missing API keys) and graceful exit
      (typing "exit", or Ctrl+C) without leaking stack traces to the user.

This file contains no business logic of its own — it only handles
input/output and delegates all real work to agent.py. Run it with:

    python app.py
"""

from __future__ import annotations

import sys

from agent import BusinessAgent

EXIT_COMMANDS = {"exit", "quit"}

BANNER = """
==================================================
  AI Business Research Agent
==================================================
Ask me about internal business data (employees,
salaries, revenue, product sales, etc.) or general
business/web research questions.

Type 'exit' to quit.
==================================================
"""


def _print_welcome() -> None:
    """Print the CLI banner shown at startup."""
    print(BANNER)


def _read_question() -> str:
    """
    Prompt the user for input and return the trimmed question.

    Returns:
        The user's input, stripped of leading/trailing whitespace.
    """
    return input("You: ").strip()


def run_cli() -> None:
    """
    Run the interactive command-line chat loop.

    Initializes the BusinessAgent once, then repeatedly reads a question
    from the user, sends it to the agent, and prints the answer, until
    the user types 'exit' (or 'quit'), presses Ctrl+C, or closes stdin
    (Ctrl+D).
    """
    _print_welcome()

    try:
        agent = BusinessAgent(verbose=False)
    except EnvironmentError as exc:
        # Missing API keys, etc. — fail fast with a clear message instead
        # of letting the user type questions that can never be answered.
        print(f"Startup failed: {exc}")
        sys.exit(1)

    print("Agent is ready. How can I help you?\n")

    while True:
        try:
            question = _read_question()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C: exit quietly and cleanly.
            print("\nGoodbye!")
            break

        if not question:
            # Ignore empty input rather than sending it to the agent.
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        answer = agent.ask(question)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    run_cli()
