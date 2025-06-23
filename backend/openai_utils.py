# backend/openai_utils.py
"""
Central place for anything that touches the OpenAI API.
Every other backend module should just:

    from backend.openai_utils import chat, MAX_PATIENTS

and never worry about keys or model names again.
"""
from __future__ import annotations
import os, openai, streamlit as st

# ------------------------------------------------------------------ secrets
# 1️⃣  Primary source: Streamlit Cloud Secrets manager
# 2️⃣  Fallback:       regular environment variables for local runs
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

MODEL        : str  = st.secrets.get("MODEL",        "gpt-4o-mini")
TEMPERATURE  : float= float(st.secrets.get("TEMPERATURE", 0.6))
MAX_PATIENTS : int  = int(st.secrets.get("MAX_PATIENTS", 500))

# ------------------------------------------------------------------ helper
def chat(messages: list[dict], **kwargs):
    """
    Thin wrapper around openai.chat.completions.create() that injects
    the default model + temperature but lets callers override anything
    via **kwargs.
    """
    return openai.chat.completions.create(
        model       = MODEL,
        temperature = TEMPERATURE,
        messages    = messages,
        **kwargs,
    )
