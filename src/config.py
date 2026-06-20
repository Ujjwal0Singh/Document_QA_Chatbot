"""
config.py
---------
Centralized configuration for the Document Q&A Bot.

All tunable constants (model names, paths, thresholds) live here so the
ingestion and query pipelines never hard-code "magic numbers" or strings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
load_dotenv()  # Reads variables from a local .env file into os.environ


def _get_secret(key: str) -> str | None:
    """
    Look up a secret in this order:
      1. Streamlit's st.secrets (only relevant if you run the optional
         bonus Streamlit UI in app.py on Streamlit Community Cloud)
      2. Environment variables / .env file (local development, CLI usage)
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # st.secrets not available - fall through to plain env vars
    return os.getenv(key)


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. Create a .env file at the project root "
        "with: GEMINI_API_KEY=\"your_actual_gemini_api_key_here\""
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
COLLECTION_NAME = "document_qa_kb"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ---------------------------------------------------------------------------
# Gemini model identifiers
# ---------------------------------------------------------------------------
# NOTE: the assignment brief references google-generativeai +
# text-embedding-004. Both are deprecated/retired by Google as of early
# 2026, so this project uses their current replacements:
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIMENSIONALITY = 768  # smaller footprint; plenty for this scale

# ---------------------------------------------------------------------------
# Chunking parameters (Step 3: Recursive Character Splitting)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # characters shared between adjacent chunks

# ---------------------------------------------------------------------------
# Retrieval parameters (Step 5)
# ---------------------------------------------------------------------------
TOP_K = 4                          # number of chunks retrieved per query
MIN_SIMILARITY_THRESHOLD = 0.3     # chunks below this cosine similarity are dropped as noise

# ---------------------------------------------------------------------------
# Embedding API batching
# ---------------------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 90  # stay safely under Gemini's per-request batch limits
