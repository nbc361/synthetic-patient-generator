"""
backend/data_ingest.py
─────────────────────────────────────────────────────────────────────────
Lightweight helper that takes Streamlit-uploaded files (PDF / DOCX / TXT)
and returns a (filename, snippet) list that downstream code can pass to an
LLM prompt or a vector-store.

The file is fully self-contained and resilient to the LangChain package-split
(works whether you have `langchain-community` installed or the older monolith).
"""

from pathlib import Path
import tempfile, shutil, os
from typing import List, Tuple

# ── 1.  Loader imports ──────────────────────────────────────────────────
# Try the new split-out package first; fall back to the legacy path so the
# app still boots if the wheel was not installed for some reason.
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )
except ModuleNotFoundError:  # pre-split LangChain (< 0.1.19)
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )


MAX_CHARS = 4_000  # cap text so we don’t blow the token budget


# ── 2.  Internal helpers ────────────────────────────────────────────────
def _loader_for(ext: str):
    """Return the appropriate LangChain loader class for a file extension."""
    ext = ext.lower()
    if ext == ".pdf":
        return PyPDFLoader
    if ext in {".txt", ".text"}:
        return TextLoader
    if ext in {".docx", ".doc"}:
        return Docx2txtLoader
    raise ValueError(f"Unsupported file type: {ext}")


# ── 3.  Public API  ─────────────────────────────────────────────────────
def ingest(uploaded_files) -> List[Tuple[str, str]]:
    """
    Parameters
    ----------
    uploaded_files : list[streamlit.UploadedFile]
        The list returned by `st.file_uploader(..., accept_multiple_files=True)`

    Returns
    -------
    list[(filename, snippet)]
        Each snippet is at most MAX_CHARS characters (≈1k tokens) – just
        enough context for an LLM prompt or quick vector-embed.
    """
    results: List[Tuple[str, str]] = []

    # Work in a temp folder so loaders can open real file paths
    workdir = Path(tempfile.mkdtemp())

    try:
        for up in uploaded_files:
            ext = Path(up.name).suffix
            tmp_path = workdir / up.name

            # Persist upload to disk
            with tmp_path.open("wb") as f:
                f.write(up.getbuffer())

            # Load & concatenate pages / paragraphs
            loader_cls = _loader_for(ext)
            docs = loader_cls(str(tmp_path)).load()
            full_text = "\n".join(d.page_content for d in docs)

            results.append((up.name, full_text[:MAX_CHARS]))

    finally:
        # Always clean up – ignore errors if the dir is already gone
        shutil.rmtree(workdir, ignore_errors=True)

    return results
