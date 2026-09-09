# 🎓 AI Student Support Assistant

An **agentic AI** system that answers college-related questions from
regulations, syllabus, FAQs, and notices — built with **RAG + Tools +
Memory**, running entirely on a **local open-source LLM via Ollama**
(no API keys, no cloud costs), orchestrated with **LangGraph** (built on
LangChain).

---

## Key Agent Capabilities

| Capability | How it's implemented |
|---|---|
| **RAG** | Documents in `data/` (regulations, syllabus, FAQs, notices) are chunked, embedded with a local Ollama embedding model, and stored in a **Chroma** vector database. The agent calls `search_knowledge_base` to retrieve grounded context before answering. |
| **Tools** | `search_knowledge_base` (RAG lookup), `calculate_attendance` (deterministic math: % + classes needed for eligibility), `get_latest_notices` (structured recent-notices lookup), `remember_this` / `recall_memory` (long-term memory tools). |
| **Memory** | **Short-term:** LangGraph's `MemorySaver` checkpointer keeps full multi-turn conversation state per session. **Long-term:** `memory_store.py` persists facts about a student (roll no., program, preferences, reminders) to a JSON file that survives across sessions. |

---

## Architecture

```
                         ┌─────────────────────────┐
                         │   Student (CLI / Web)   │
                         └────────────┬────────────┘
                                      │ question
                                      ▼
                         ┌─────────────────────────┐
                         │   LangGraph ReAct Agent │
                         │   (agent.py)             │
                         │   LLM: Ollama (llama3.1) │
                         └────────────┬────────────┘
                                      │ decides which tool(s) to call
                       ┌──────────────┼───────────────┬───────────────┐
                       ▼              ▼               ▼               ▼
              search_knowledge_base  calculate_    get_latest_   remember_this /
              (RAG / Chroma DB)      attendance    notices       recall_memory
                       │                                              │
                       ▼                                              ▼
              data/*.md  (regulations,                      memory_data/*.json
              syllabus, faqs, notices)                       (per-student facts)
```

- **Short-term memory** = LangGraph checkpointer (in-process, per conversation thread)
- **Long-term memory** = `memory_store.py` (on-disk, per student, across sessions)

---

## Project Structure

```
student_support_assistant/
├── data/
│   ├── regulations.md      # attendance, exams, grading, fees, conduct rules
│   ├── syllabus.md         # sample Semester 5 CSE syllabus
│   ├── faqs.md             # common administrative Q&A
│   └── notices.md          # sample official notices/announcements
├── memory_data/            # auto-created: per-student long-term memory (JSON)
├── chroma_db/              # auto-created by ingest.py: the vector store
├── ingest.py                # builds/rebuilds the RAG vector store
├── tools.py                  # RAG tool + attendance calculator + notices + memory tools
├── memory_store.py            # simple JSON-based long-term memory backend
├── agent.py                    # LangGraph agent definition (LLM + tools + checkpointer)
├── main.py                      # CLI chat interface
├── app.py                        # optional Streamlit web chat UI
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Ollama and pull models

Download Ollama from **https://ollama.com** for your OS, then:

```bash
# Chat model (any of these work — pick based on your hardware)
ollama pull llama3.1        # good default, 8B
# ollama pull llama3        # alternative
# ollama pull qwen2.5       # alternative, strong tool-calling support
# ollama pull mistral       # lighter alternative

# Embedding model for RAG
ollama pull nomic-embed-text
```

Make sure the Ollama server is running:
```bash
ollama serve
```
(On macOS/Windows the desktop app runs this automatically.)

### 2. Install Python dependencies

```bash
cd student_support_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the RAG vector store

```bash
python ingest.py
```

This reads everything in `data/`, chunks it, embeds it with
`nomic-embed-text`, and saves a Chroma DB to `./chroma_db`. Re-run this
any time you edit or add documents in `data/`.

### 4. Run the assistant

**CLI:**
```bash
python main.py
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

---

## Example Questions to Try

- "What is the minimum attendance required to write exams?"
- "I've attended 60 out of 90 classes so far — am I eligible for the exam?"
- "What topics are covered in the AI syllabus this semester?"
- "How do I apply for revaluation?"
- "Any recent notices I should know about?"
- "Remember that my roll number is CSE21045 and I'm in Section B."
- (later in conversation) "What's my roll number again?"

---

## Configuration

Environment variables (optional, all have sensible defaults):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `llama3.1` | Ollama chat model name |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

Example:
```bash
LLM_MODEL=qwen2.5 python main.py
```

---

## Extending This Project

- **Add more documents:** drop more `.md`/`.txt` files into `data/`, update
  `CATEGORY_MAP` in `ingest.py`, and re-run `python ingest.py`.
- **Swap the vector DB:** replace Chroma with FAISS/Pinecone/Weaviate by
  editing `tools.py` and `ingest.py` — the rest of the agent is unaffected.
- **Swap the LLM:** point `LLM_MODEL` at any tool-calling-capable model
  pulled into Ollama, or swap `ChatOllama` for another LangChain chat
  model class entirely (e.g. a hosted API) in `agent.py`.
- **Persist short-term memory across restarts:** replace `MemorySaver()`
  in `agent.py` with LangGraph's `SqliteSaver` or `PostgresSaver`.
- **Add more tools:** e.g. a fee-due-date reminder, a GPA calculator, or
  a connector to a real Student Information System API — add a new
  `@tool`-decorated function in `tools.py` and append it to `ALL_TOOLS`.
