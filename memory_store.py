"""
memory_store.py
----------------
A tiny, dependency-free long-term memory store for the agent.

This is separate from LangGraph's checkpointer (which only remembers the
back-and-forth of the *current conversation thread*). This module lets the
agent persist facts about a student ACROSS sessions -- e.g. their roll
number, program, or a reminder they asked to be saved -- by writing to a
JSON file on disk.

In a production system you'd swap this for a real database (Postgres,
Redis, etc.), but the interface (save_fact / search_facts) would stay the
same, which keeps the agent tools unchanged.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

STORE_DIR = Path(__file__).parent / "memory_data"
STORE_DIR.mkdir(exist_ok=True)


def _store_path(student_id: str) -> Path:
    safe_id = "".join(c for c in student_id if c.isalnum() or c in ("-", "_")) or "default"
    return STORE_DIR / f"{safe_id}.json"


def _load(student_id: str) -> List[Dict]:
    path = _store_path(student_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(student_id: str, facts: List[Dict]):
    path = _store_path(student_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def save_fact(student_id: str, fact: str) -> str:
    """Append a new remembered fact for this student."""
    facts = _load(student_id)
    facts.append({"fact": fact, "saved_at": datetime.now().isoformat(timespec="seconds")})
    _save(student_id, facts)
    return f"Saved to memory: '{fact}'"


def search_facts(student_id: str, query: str = "") -> str:
    """Return remembered facts for this student, optionally filtered by a keyword."""
    facts = _load(student_id)
    if not facts:
        return "No saved memory found for this student yet."

    if query:
        query_lower = query.lower()
        matches = [f for f in facts if query_lower in f["fact"].lower()]
    else:
        matches = facts

    if not matches:
        return f"No saved memory matching '{query}' found."

    lines = [f"- {f['fact']} (saved {f['saved_at']})" for f in matches]
    return "\n".join(lines)


def list_all_facts(student_id: str) -> List[Dict]:
    return _load(student_id)
