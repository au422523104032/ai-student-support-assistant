"""
agent.py
--------
Defines the AI Student Support Assistant as a LangGraph ReAct-style agent.

Agentic capabilities wired up here:
- RAG      -> search_knowledge_base tool (Chroma vector store, see tools.py)
- Tools    -> calculate_attendance, get_latest_notices, remember_this, recall_memory
- Memory   -> short-term: LangGraph's MemorySaver checkpointer keeps full
              conversation history per thread_id (multi-turn context).
              long-term: memory_store.py persists facts about a student to
              disk across separate sessions (see remember_this/recall_memory).

The LLM itself runs fully locally through Ollama (e.g. llama3, llama3.1,
qwen2.5, mistral, etc. - anything you've pulled with `ollama pull`).
"""

import os

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS

LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are the AI Student Support Assistant for a college.

Your job is to help students with questions about:
- Academic regulations (attendance, exams, grading, revaluation, fees, conduct)
- Course syllabus and subject details
- Frequently asked administrative questions
- Official notices and announcements

Guidelines:
1. For ANY factual question about college rules, courses, procedures, or
   announcements, ALWAYS call the `search_knowledge_base` tool first rather
   than answering from your own general knowledge. Ground your answer in
   what the tool returns, and mention which category the information came
   from (e.g. "According to the Academic Regulations...").
2. If a student asks about their attendance percentage or eligibility,
   use the `calculate_attendance` tool rather than doing the math yourself.
3. If a student asks what's new or about recent announcements, use
   `get_latest_notices`.
4. If the student shares something worth remembering for later (their
   roll number, program, a reminder, a preference), call `remember_this`.
   If they refer to something from a past conversation, call
   `recall_memory` to check saved facts before saying you don't know.
5. If the knowledge base does not contain the answer, say so honestly and
   suggest which office/department the student should contact, rather
   than inventing a policy.
6. Be concise, warm, and precise. Cite specific numbers/percentages/dates
   from the tools rather than vague summaries when they are available.
7. You will be given a `student_id` for the current session at the start
   of the conversation context - use that exact value whenever a tool
   requires student_id.
"""


def build_agent():
    """Construct and return a compiled LangGraph agent with memory."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    checkpointer = MemorySaver()  # short-term / in-conversation memory

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
