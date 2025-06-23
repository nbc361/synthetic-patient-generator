# backend/ingest.py
"""
Turn user-supplied documents into a Chroma vector-store we can query later.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter   import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores     import Chroma

from backend.openai_utils import MODEL          # re-use model family
# ---------------------------------------------------------------------------

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size   = 1_000,
    chunk_overlap= 200,
    separators   = ["\n\n", "\n", " ", ""],
)

_EMBEDDER = OpenAIEmbeddings(model=MODEL)

def _load(path: Path):
    """Return a LangChain Document list for a single file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".docx", ".doc"}:
        return Docx2txtLoader(str(path)).load()
    if suffix == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}")

def ingest(files: Iterable, scope_notes: list[str]) -> Chroma:
    """
    • `files` – list of `UploadedFile` objects from Streamlit  
    • `scope_notes` – list of one-line descriptions the user typed

    Returns an **in-memory Chroma** index you can immediately `.sim_search()`.
    """
    all_docs = []
    for up, note in zip(files, scope_notes):
        # Streamlit’s UploadedFile -> temp file on disk
        tmp_path = Path(up.name)
        with tmp_path.open("wb") as f:
            f.write(up.read())

        docs = _load(tmp_path)
        for d in docs:
            d.metadata.setdefault("scope_note", note)
            d.metadata.setdefault("source_file", up.name)
        all_docs.extend(docs)
        tmp_path.unlink(missing_ok=True)  # tidy up

    # ---- split & embed
    chunks = _SPLITTER.split_documents(all_docs)
    vectordb = Chroma.from_documents(chunks, _EMBEDDER)  # in-memory
    return vectordb
