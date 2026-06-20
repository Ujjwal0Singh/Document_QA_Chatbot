"""
main.py
-------
User interface: an interactive command-line loop for the Document Q&A Bot.

Loads the pre-existing vector database (built by ingest.py), then repeatedly
takes user questions, retrieves relevant chunks, and prints a grounded,
cited answer.

Run with:
    python -m src.main
"""

import logging

from src import config, query

logging.basicConfig(level=logging.WARNING)  # keep the CLI output clean
logger = logging.getLogger(__name__)


def _print_banner():
    print("=" * 60)
    print(" Document Q&A Bot (RAG-powered)")
    print(" Ask a question about the documents in data/.")
    print(" Type 'exit' or 'quit' to leave.")
    print("=" * 60)


def _print_sources(chunks):
    if not chunks:
        print("\n(No documents were relevant enough to cite.)")
        return
    print("\nSources used:")
    seen = set()
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            print(f"  - {chunk.source}, Page {chunk.page}  (similarity: {chunk.score:.2f})")


def main():
    _print_banner()

    try:
        collection = query.load_vector_store()
    except RuntimeError as exc:
        print(f"\n⚠️  {exc}")
        return

    print(f"\nLoaded vector database with {collection.count()} indexed chunks.\n")

    while True:
        try:
            user_question = input("Your question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = query.answer_question(user_question, collection=collection)

        print("\n" + "-" * 60)
        print(result.answer)
        _print_sources(result.retrieved_chunks)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
