# backend/data_ingest.py
"""
Very small helper: turn a list[UploadedFile] from Streamlit
into a Chroma VectorStore we can query with similarity_search().
"""

from pathlib import Path
import tempfile, shutil, os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter   # <- stays the same

# one global embedding model (uses OPENAI_API_KEY from env / st.secrets)
_EMBED = OpenAIEmbeddings()

# splitter keeps chunks ≤ 1 k tokens and ~15 % overlap for recall
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size       = 1_000,
    chunk_overlap    = 150,
    length_function  = len,
)

def _tmp_copy(uploaded_file):
    """Save Streamlit's in-memory file to a real path and return it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
    tmp.write(uploaded_file.read())
    tmp.flush()
    return Path(tmp.name)

# ───────────────────────────────────────────────────────────────────────────
def ingest(files, notes=None):
    """
    • files  – list[streamlit.runtime.uploaded_file_manager.UploadedFile]
    • notes  – optional list[str] human labels (same order)

    Returns an IN-MEMORY Chroma VectorStore.
    """
    docs = []

    for idx, up in enumerate(files):
        label = (notes[idx] if notes and idx < len(notes) else up.name).strip() or f"doc_{idx}"
        local = _tmp_copy(up)

        # choose loader based on file suffix
        suffix = local.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(local))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(local))
        else:
            loader = TextLoader(str(local), autodetect_encoding=True)

        for doc in loader.load_and_split(_SPLITTER):
            # tack on the friendly label so we can interpret results later
            doc.metadata.setdefault("source", label)
            docs.append(doc)

        # clean up the tmp file – Chroma only needs text & embeddings now
        try: os.remove(local)
        except OSError: pass

    if not docs:
        raise ValueError("No parsable content found in uploaded files.")

    vectordb = Chroma.from_documents(
        docs,
        embedding=_EMBED,
        collection_name="ephemeral-files",    # kept only for this request
        persist_directory=None,               # -> in memory
    )
    return vectordb
