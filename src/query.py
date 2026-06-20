"""
query.py
--------
Query pipeline: loads the pre-existing ChromaDB vector store from disk,
embeds the user's question, retrieves the top-k most relevant chunks,
and asks Gemini to answer strictly from that retrieved context, with
citations.
"""

import os

# Must be set BEFORE chromadb is imported anywhere in the process - avoids a
# protobuf/opentelemetry descriptor TypeError on some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import logging
from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher = more relevant


@dataclass
class QAResult:
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    has_grounded_context: bool


_SYSTEM_PROMPT = """\
You are a precise document Q&A assistant. Use ONLY the provided context to \
answer the user's question. If the answer cannot be found in the context, \
say "I cannot find the answer in the provided documents." Do not attempt \
to use your own knowledge to answer.

Whenever you state a fact drawn from the context, cite its source \
immediately after the sentence in the format (filename, Page X), matching \
the source and page labels given in the context below.

CONTEXT:
{context}
"""


# ---------------------------------------------------------------------------
# Loading the persisted vector store
# ---------------------------------------------------------------------------
def load_vector_store():
    """Load the existing ChromaDB collection from disk (no re-embedding)."""
    db_client = chromadb.PersistentClient(
        path=str(config.DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = db_client.get_collection(config.COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "No existing vector database found. Run 'python -m src.ingest' "
            "first to index your documents."
        ) from exc

    if collection.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Run 'python -m src.ingest' "
            "to index documents in data/ before asking questions."
        )

    return collection


# ---------------------------------------------------------------------------
# Step 5: Similarity Search & Retrieval
# ---------------------------------------------------------------------------
def embed_query(query: str) -> List[float]:
    """Embed the user's question using the exact same embedding model/config
    used to index the document chunks, so the vectors are comparable."""
    result = _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=config.EMBEDDING_OUTPUT_DIMENSIONALITY,
        ),
    )
    return result.embeddings[0].values


def retrieve_relevant_chunks(query: str, collection, top_k: int = config.TOP_K) -> List[RetrievedChunk]:
    """
    Embed the query, fetch the top-k closest chunks by cosine similarity,
    and drop anything below MIN_SIMILARITY_THRESHOLD so unrelated noise
    never reaches the prompt.
    """
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]  # Chroma returns cosine *distance*

    retrieved = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - distance
        if similarity >= config.MIN_SIMILARITY_THRESHOLD:
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", 0),
                    score=similarity,
                )
            )

    return retrieved


# ---------------------------------------------------------------------------
# Step 6: Prompt Engineering & Answer Generation
# ---------------------------------------------------------------------------
def _format_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant content was found in the document library.)"

    sections = []
    for chunk in chunks:
        sections.append(
            f'[Source: {chunk.source}, Page: {chunk.page}]\n"{chunk.text}"'
        )
    return "\n\n".join(sections)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Ask Gemini to answer strictly from the retrieved, cited context."""
    system_prompt = _SYSTEM_PROMPT.format(context=_format_context(chunks))

    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        )
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Generation call failed: %s", exc)
        return "Sorry, I ran into a technical issue while generating an answer. Please try again."


def answer_question(query: str, collection=None) -> QAResult:
    """
    End-to-end: retrieve relevant chunks for `query`, then generate a
    grounded, cited answer. Loads the collection itself if not provided.
    """
    if collection is None:
        collection = load_vector_store()

    chunks = retrieve_relevant_chunks(query, collection)
    answer = generate_answer(query, chunks)

    return QAResult(
        answer=answer,
        retrieved_chunks=chunks,
        has_grounded_context=len(chunks) > 0,
    )
