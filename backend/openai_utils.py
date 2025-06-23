# backend/openai_utils.py
import streamlit as st
import openai, os

# Grab values from Streamlit Cloud's Secrets manager
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Optional: expose a configured client + some constants
MODEL         = st.secrets.get("MODEL",         "gpt-4o-mini")
TEMPERATURE   = st.secrets.get("TEMPERATURE",   0.6)
MAX_PATIENTS  = st.secrets.get("MAX_PATIENTS",  500)

def chat(messages, **kwargs):
    """Simple wrapper around openai.chat.completions.create()"""
    return openai.chat.completions.create(
        model       = MODEL,
        temperature = TEMPERATURE,
        messages    = messages,
        **kwargs,
    )
