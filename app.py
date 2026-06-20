"""
app.py
------
Optional bonus: a simple Streamlit web UI for the Document Q&A Bot.

On first load, if no vector database exists yet (e.g. a fresh deploy on
Streamlit Cloud, where db/ is git-ignored and never gets pushed), this
automatically runs ingestion against whatever documents are in data/.
On later runs/reruns, it just loads the existing database from disk.

Run with:
    streamlit run app.py
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st

from src import ingest, query

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading vector database...")
def get_collection():
    try:
        return query.load_vector_store()
    except RuntimeError:
        # No index yet (fresh deploy) - build it now from data/.
        with st.spinner("No index found yet - building it from data/ (first run only)..."):
            ingest.run_ingestion()
        return query.load_vector_store()


def main():
    st.title("📄 Document Q&A Bot")
    st.caption("Ask questions about the documents in data/. Answers are grounded and cited.")

    try:
        collection = get_collection()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Vector database loaded — {collection.count()} indexed chunks.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.write(f"- {source}")

    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = query.answer_question(user_question, collection=collection)
            st.markdown(result.answer)

            sources = sorted({
                f"{c.source}, Page {c.page} (similarity: {c.score:.2f})"
                for c in result.retrieved_chunks
            })
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer, "sources": sources}
        )


if __name__ == "__main__":
    main()