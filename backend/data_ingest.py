"""
backend/ingest.py
Turns uploaded PDFs, DOCX, or TXT files into
1 500-character text chunks, and attaches the user’s
per-file “scope note” comment to every chunk.
"""

from pathlib import Path
from typing import List, Tuple
import re

import PyPDF2
import docx  # from python-docx

# ─────────────────────────────────────────────────────────
# helper – split long text into ~1 500-char overlapping chunks
# ─────────────────────────────────────────────────────────
def _split_text(text: str, size: int = 1500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # small overlap so sentences don’t break
    return chunks


# ─────────────────────────────────────────────────────────
# main entry
# files  : list of Streamlit UploadedFile
# notes  : list of strings (one per file, already validated)
# returns: list[Tuple[str, str]]  -> (chunk_text, scope_note)
# ─────────────────────────────────────────────────────────
def chunk_docs(files, notes: List[str]) -> List[Tuple[str, str]]:
    out_chunks = []

    for file_obj, note in zip(files, notes):
        suffix = Path(file_obj.name).suffix.lower()

        # --- read raw full text from each file type -----------------------
        if suffix == ".pdf":
            reader = PyPDF2.PdfReader(file_obj)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif suffix == ".docx":
            doc = docx.Document(file_obj)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif suffix == ".txt":
            text = file_obj.read().decode("utf-8", errors="ignore")

        else:
            # should never happen (we limited types in uploader)
            text = ""

        # tidy whitespace
        text = re.sub(r"\s+\n", "\n", text)        # rm long spaces before linebreak
        text = re.sub(r"\n\s+", "\n", text)        # rm indents
        text = re.sub(r"\n{3,}", "\n\n", text)     # squash >2 blank lines

        # --- split into chunks and attach scope note ----------------------
        for chunk in _split_text(text):
            if chunk.strip():                      # skip empty
                out_chunks.append((chunk, note.strip()))

    return out_chunks
