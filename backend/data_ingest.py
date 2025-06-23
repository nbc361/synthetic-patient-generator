# backend/data_ingest.py
"""
Save the Streamlit-uploaded docs to a temp dir, load them with
LangChain loaders, then build an in-memory Chroma vector-DB.
"""

from __future__ import annotations
from pathlib import Path
import tempfile, shutil, os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------
def _to_temp_file(uploaded, out_dir: Path) -> Path:
    """Persist a Streamlit UploadedFile to disk and return the path."""
    out_path = out_dir / uploaded.name
    with out_path.open("wb") as f:
        f.write(uploaded.getbuffer())
    return out_path


def ingest(files, comments: list[str]):
    """
    • `files`   → list[streamlit.UploadedFile]  
    • `comments`→ list[str] (same length as files)

    Returns a Chroma vector-store you can `.similarity_search()`.
    """

    if not files:
        raise ValueError("No files passed to ingest()")

    # temp workspace
    tmp_root = Path(tempfile.mkdtemp())
    docs = []

    for idx, up in enumerate(files):
        note = comments[idx] if idx < len(comments) else up.name
        path = _to_temp_file(up, tmp_root)

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".txt", ".text"}:
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(f"Unsupported file type: {path.name}")

        for d in loader.load():
            d.metadata["source_note"] = note
            docs.append(d)

    # build the mini vector-db (Ephemeral / in-memory)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=None,   # purely in RAM
    )

    # clean temp files when the process exits
    shutil.rmtree(tmp_root, ignore_errors=True)
    return vectordb
