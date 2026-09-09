"""
tools.py
--------
All tools exposed to the agent:

1. search_knowledge_base  -> RAG retrieval over regulations/syllabus/FAQs/notices
2. calculate_attendance   -> deterministic attendance % / classes-needed calculator
3. get_latest_notices     -> quick structured lookup of the most recent notices
4. remember_this          -> long-term memory: save a fact about the student
5. recall_memory          -> long-term memory: retrieve saved facts

Keeping tools as small, single-purpose functions (rather than one giant
tool) lets the LLM pick the right one and keeps each tool's output
predictable and easy to test.
"""

import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import memory_store

PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "student_support_kb"

_vectorstore = None


def _get_vectorstore():
    """Lazy-load the Chroma vector store (built by ingest.py)."""
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(PERSIST_DIR):
            raise RuntimeError(
                "Vector store not found. Run `python ingest.py` first to index "
                "the documents in the data/ folder."
            )
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the college's knowledge base (academic regulations, syllabus,
    FAQs, and official notices) for information relevant to the query.
    Use this for ANY question about college rules, courses, procedures,
    or announcements. Returns the most relevant passages along with which
    document category each came from.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=4)
    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, doc in enumerate(results, 1):
        category = doc.metadata.get("category", "General")
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"[Result {i} | Source: {category} ({source})]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


@tool
def calculate_attendance(
    classes_attended: int, classes_held: int, target_percentage: float = 75.0
) -> str:
    """
    Calculate a student's current attendance percentage and, if it is
    below the target (default 75%, per college regulations), how many
    additional consecutive classes they would need to attend to reach it.
    Use this whenever a student asks about their attendance percentage,
    whether they'll be eligible for exams, or how many classes they can
    afford to miss.
    """
    if classes_held <= 0:
        return "classes_held must be greater than 0."
    if classes_attended < 0 or classes_attended > classes_held:
        return "classes_attended must be between 0 and classes_held."

    current_pct = (classes_attended / classes_held) * 100
    result = [f"Current attendance: {classes_attended}/{classes_held} = {current_pct:.2f}%"]

    if current_pct >= target_percentage:
        # How many classes can the student afford to miss and stay at/above target?
        max_holdable = classes_attended / (target_percentage / 100)
        can_miss = int(max_holdable) - classes_held
        result.append(
            f"You are at or above the {target_percentage:.0f}% requirement. "
            f"You can miss approximately {max(can_miss, 0)} more class(es) "
            f"(assuming no further classes are added beyond that) and stay compliant."
        )
    else:
        # Solve for x additional classes attended (and held) needed to hit target.
        # (attended + x) / (held + x) >= target/100
        t = target_percentage / 100
        if t >= 1:
            result.append("Target percentage must be below 100% to compute a solution.")
        else:
            x = (t * classes_held - classes_attended) / (1 - t)
            x = max(0, int(x) + 1)
            result.append(
                f"You are below the {target_percentage:.0f}% requirement. "
                f"You need to attend approximately {x} more consecutive class(es) "
                f"(with no further absences) to reach {target_percentage:.0f}%."
            )
        if current_pct < 65:
            result.append(
                "Note: per college regulations, attendance below 65% means you "
                "will be DETAINED from semester exams, with no condonation option."
            )
        elif current_pct < 75:
            result.append(
                "Note: per college regulations, attendance between 65-74% may be "
                "eligible for condonation via the HOD with a valid medical certificate."
            )

    return "\n".join(result)


@tool
def get_latest_notices(count: int = 3) -> str:
    """
    Return the most recent official college notices (announcements about
    exams, fees, events, placements, etc.), most recent first. Use this
    when a student asks 'what's new', 'any recent notices', or about
    upcoming deadlines/events without a specific topic in mind.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(
        "official notice announcement", k=count, filter={"category": "Notice"}
    )
    if not results:
        return "No notices found."
    formatted = [f"- {doc.page_content.strip()}" for doc in results[:count]]
    return "\n\n".join(formatted)


@tool
def remember_this(student_id: str, fact: str) -> str:
    """
    Save a piece of information the student wants remembered for future
    conversations (e.g. their roll number, program, a personal reminder,
    or a stated preference). Use this only when the student explicitly
    shares something they want remembered, or asks you to remember it.
    """
    return memory_store.save_fact(student_id, fact)


@tool
def recall_memory(student_id: str, query: str = "") -> str:
    """
    Retrieve previously saved facts about this student (from earlier
    sessions), optionally filtered by a keyword. Use this when the
    student refers to something they told you before, or asks 'what do
    you remember about me'.
    """
    return memory_store.search_facts(student_id, query)


ALL_TOOLS = [
    search_knowledge_base,
    calculate_attendance,
    get_latest_notices,
    remember_this,
    recall_memory,
]
