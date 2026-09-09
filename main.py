"""
main.py
-------
Command-line chat interface for the AI Student Support Assistant.

Run:
    python main.py

Make sure you have already run `python ingest.py` at least once, and that
Ollama is running locally with the required models pulled (see README.md).
"""

import sys
import uuid

from langchain_core.messages import HumanMessage

from agent import build_agent


def main():
    print("=" * 60)
    print(" AI Student Support Assistant  (RAG + Tools + Memory)")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=" * 60)

    student_id = input("\nEnter your student ID (used to save your memory): ").strip()
    if not student_id:
        student_id = "guest"

    # thread_id scopes LangGraph's short-term conversation memory.
    # Using the student_id means resuming won't happen automatically between
    # separate `python main.py` runs unless you persist the checkpointer to
    # disk (MemorySaver here is in-memory only, by design for this demo).
    thread_id = f"{student_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    agent = build_agent()

    print(f"\nHi {student_id}! Ask me about regulations, syllabus, FAQs, or notices.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Prefix the student_id into context so the agent can pass it to
        # memory tools without asking the student to repeat it every time.
        contextualized_input = f"[student_id={student_id}] {user_input}"

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=contextualized_input)]},
                config=config,
            )
            final_message = result["messages"][-1]
            print(f"\nAssistant: {final_message.content}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")
            print(
                "Tip: make sure Ollama is running (`ollama serve`) and that you've "
                "pulled the required models (see README.md).\n"
            )


if __name__ == "__main__":
    sys.exit(main())
