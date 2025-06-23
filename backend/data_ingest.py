# backend/data_ingest.py
"""
Turn Streamlit-uploaded docs into an in-memory FAISS VectorStore.
FAISS has no server and works on Streamlit Cloud.
"""

from __future__ import annotations
from pathlib import Path
import tempfile, shutil, os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings


# ----------------------------------------------------------------------
def _save_uploaded(uploaded, out_dir: Path) -> Path:
    path = out_dir / uploaded.name
    with path.open("wb") as f:
        f.write(uploaded.getbuffer())
    return path


def ingest(files, comments: list[str]):
    """Return a FAISS vectorstore ready for `.similarity_search()`."""
    if not files:
        raise ValueError("No files to ingest")

    tmp = Path(tempfile.mkdtemp())
    docs = []

    for i, up in enumerate(files):
        note = comments[i] if i < len(comments) else up.name
        p = _save_uploaded(up, tmp)

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(p))
        elif suffix in {".txt", ".text"}:
            loader = TextLoader(str(p), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {p.name}")

        for d in loader.load():
            d.metadata["source_note"] = note
            docs.append(d)

    vectordb = FAISS.from_documents(docs, OpenAIEmbeddings())

    # tidy temp dir when process ends
    shutil.rmtree(tmp, ignore_errors=True)
    return vectordb
