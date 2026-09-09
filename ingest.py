"""
ingest.py
---------
Builds (or rebuilds) the local Chroma vector store used for RAG.

Reads every .md / .txt file inside ./data (regulations, syllabus, FAQs,
notices), splits them into chunks, embeds them with a local Ollama
embedding model, and persists them to ./chroma_db.

Run this once before starting the agent, and again any time you add or
change documents in the data/ folder.

Usage:
    python ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "student_support_kb"

# Map filename -> a friendly "category" used as metadata so the agent can
# tell the student which type of document an answer came from.
CATEGORY_MAP = {
    "regulations.md": "Academic Regulations",
    "syllabus.md": "Syllabus",
    "faqs.md": "FAQ",
    "notices.md": "Notice",
}


def load_documents():
    docs = []
    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found at {DATA_DIR}")
        sys.exit(1)

    files = sorted(list(DATA_DIR.glob("*.md")) + list(DATA_DIR.glob("*.txt")))
    if not files:
        print(f"ERROR: no .md or .txt files found in {DATA_DIR}")
        sys.exit(1)

    for file_path in files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        file_docs = loader.load()
        category = CATEGORY_MAP.get(file_path.name, "General")
        for d in file_docs:
            d.metadata["category"] = category
            d.metadata["source"] = file_path.name
        docs.extend(file_docs)
        print(f"  loaded {file_path.name} -> category='{category}'")
    return docs


def main():
    print(f"Loading documents from {DATA_DIR} ...")
    raw_docs = load_documents()

    print("Splitting documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  created {len(chunks)} chunks")

    print(f"Embedding with Ollama model '{EMBED_MODEL}' and building Chroma store ...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Wipe any previous store so re-running this script is idempotent.
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"\nDone. Vector store persisted at: {PERSIST_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
