"""
ingest.py
---------
Pipeline script: reads source documents from data/, extracts text with
page-level metadata, chunks the text, embeds the chunks, and persists
everything to a local ChromaDB collection on disk (db/).

Run this whenever you add or change files in data/:

    python -m src.ingest
"""

import os

# Must be set BEFORE chromadb is imported anywhere in the process - avoids a
# protobuf/opentelemetry descriptor TypeError on some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import logging
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from docx import Document as DocxDocument
from google import genai
from google.genai import types
from pypdf import PdfReader
from tqdm import tqdm

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Step 2: Document Ingestion & Text Extraction
# ---------------------------------------------------------------------------
def extract_pdf_pages(file_path: Path) -> List[Dict]:
    """Extract text page-by-page from a PDF, tagging each page with metadata."""
    pages = []
    reader = PdfReader(str(file_path))
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = " ".join(raw_text.split())  # strip excess whitespace/newlines
        if cleaned:
            pages.append({"text": cleaned, "source": file_path.name, "page": page_number})
    return pages


def extract_docx_pages(file_path: Path) -> List[Dict]:
    """
    Extract text from a .docx file. Word documents don't have a native
    'page' concept in the underlying XML, so we use the paragraph index as
    a stable position marker instead, grouping paragraphs into pseudo-pages
    of ~1 page worth of content for citation purposes.
    """
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    pseudo_pages = []
    paragraphs_per_page = 12  # rough heuristic so citations stay meaningful
    for i in range(0, len(paragraphs), paragraphs_per_page):
        page_paragraphs = paragraphs[i : i + paragraphs_per_page]
        cleaned = " ".join(" ".join(page_paragraphs).split())
        if cleaned:
            pseudo_pages.append(
                {"text": cleaned, "source": file_path.name, "page": (i // paragraphs_per_page) + 1}
            )
    return pseudo_pages


def extract_txt_pages(file_path: Path) -> List[Dict]:
    """Plain text files have no pages; tagged as page 1 for citation consistency."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned = " ".join(raw_text.split())
    if not cleaned:
        return []
    return [{"text": cleaned, "source": file_path.name, "page": 1}]


def extract_document(file_path: Path) -> List[Dict]:
    """Dispatch to the correct extractor based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    if suffix == ".docx":
        return extract_docx_pages(file_path)
    if suffix == ".txt":
        return extract_txt_pages(file_path)
    raise ValueError(f"Unsupported file type: {file_path.name}")


def load_all_documents(data_dir: Path = config.DATA_DIR) -> List[Dict]:
    """Scan data_dir for supported files and extract all pages/pseudo-pages."""
    all_pages: List[Dict] = []
    files = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported documents found in %s", data_dir)
        return all_pages

    for file_path in tqdm(files, desc="Extracting documents"):
        try:
            pages = extract_document(file_path)
            all_pages.extend(pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to extraction error: %s", file_path.name, exc)

    return all_pages


# ---------------------------------------------------------------------------
# Step 3: Text Chunking Strategy (Recursive Character Splitting)
# ---------------------------------------------------------------------------
_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    """
    A small, dependency-free recursive character splitter: tries the first
    separator; if a resulting piece is still too long, it recurses with the
    next separator down the list, finally falling back to a hard character
    cut. This mirrors LangChain's RecursiveCharacterTextSplitter behavior
    without requiring the extra package.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if separator:
        parts = text.split(separator)
    else:
        parts = list(text)  # last resort: split into individual characters

    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = current + (separator if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size and remaining_separators:
                chunks.extend(_split_text(part, chunk_size, chunk_overlap, remaining_separators))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    # Apply overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + chunks[i])
        return overlapped

    return chunks


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Chunk every page's text while preserving source/page metadata on each
    resulting chunk, so citations remain accurate after splitting.
    """
    all_chunks: List[Dict] = []
    for page in tqdm(pages, desc="Chunking pages"):
        text_chunks = _split_text(
            page["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
        )
        for chunk_text in text_chunks:
            all_chunks.append(
                {"text": chunk_text, "source": page["source"], "page": page["page"]}
            )
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed a list of texts in batches using Gemini's embedding model."""
    all_embeddings: List[List[float]] = []
    batch_size = config.EMBEDDING_BATCH_SIZE

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        result = _client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=config.EMBEDDING_OUTPUT_DIMENSIONALITY,
            ),
        )
        all_embeddings.extend(embedding.values for embedding in result.embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Step 4: Persisting the Vector Database
# ---------------------------------------------------------------------------
def build_vector_store(chunks: List[Dict]) -> None:
    """Embed all chunks and persist them, with metadata, to ChromaDB on disk."""
    if not chunks:
        logger.warning("No chunks to index; nothing was written to the database.")
        return

    db_client = chromadb.PersistentClient(
        path=str(config.DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Start fresh each time ingest.py runs, so stale chunks never linger.
    try:
        db_client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet - nothing to delete

    collection = db_client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    ids = [f"{c['source']}-p{c['page']}-{i}" for i, c in enumerate(chunks)]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    # Chroma's add() has its own internal batch limits; insert in slices.
    insert_batch_size = 200
    for i in tqdm(range(0, len(chunks), insert_batch_size), desc="Writing to ChromaDB"):
        collection.add(
            ids=ids[i : i + insert_batch_size],
            documents=texts[i : i + insert_batch_size],
            embeddings=embeddings[i : i + insert_batch_size],
            metadatas=metadatas[i : i + insert_batch_size],
        )

    logger.info(
        "Indexed %d chunks from %d source pages into '%s' (persisted at %s).",
        len(chunks), len(chunks), config.COLLECTION_NAME, config.DB_DIR,
    )


def run_ingestion() -> None:
    """End-to-end ingestion: extract -> chunk -> embed -> persist."""
    logger.info("Scanning %s for documents...", config.DATA_DIR)
    pages = load_all_documents()
    logger.info("Extracted %d pages/sections from source documents.", len(pages))

    chunks = chunk_pages(pages)
    logger.info("Produced %d chunks (chunk_size=%d, overlap=%d).",
                len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    build_vector_store(chunks)
    logger.info("Ingestion complete. Run 'python -m src.main' to start asking questions.")


if __name__ == "__main__":
    run_ingestion()
