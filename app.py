"""
StudyLens AI -- Streamlit entry point.

This file is intentionally thin: it only handles UI rendering and session
state. All real logic (PDF parsing, chunking, embeddings, retrieval, LLM
calls) lives in `src/`, so this file stays readable and the pipeline stays
testable without Streamlit.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.generation.llm import LLMClient, LLMError
from src.ingestion.pdf_loader import PDFLoadError
from src.knowledge_base import KnowledgeBase
from src.retrieval.embeddings import EmbeddingError, get_embedding_model
from src.retrieval.vector_store import VectorStoreError
from src.utils.config import ConfigError, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("studylens")

st.set_page_config(
    page_title="StudyLens AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 920px;
    }

    .sl-header {
        margin-bottom: 0.25rem;
    }
    .sl-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sl-tagline {
        color: rgba(120, 120, 130, 0.9);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .sl-brand {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }
    .sl-brand-sub {
        font-size: 0.8rem;
        color: rgba(120, 120, 130, 0.85);
        margin-bottom: 1.2rem;
    }

    .sl-section-label {
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: rgba(130, 130, 140, 0.9);
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
    }

    .sl-stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        padding: 0.15rem 0;
    }
    .sl-stat-label { color: rgba(140, 140, 150, 0.95); }
    .sl-stat-value { font-weight: 600; }

    .sl-doc-chip {
        font-size: 0.82rem;
        padding: 0.35rem 0.6rem;
        border-radius: 8px;
        background: rgba(130, 130, 160, 0.08);
        margin-bottom: 0.35rem;
        overflow-wrap: break-word;
    }

    .sl-empty-state {
        text-align: center;
        padding: 3.5rem 1.5rem;
        border: 1.5px dashed rgba(140, 140, 160, 0.28);
        border-radius: 16px;
        margin-top: 1.5rem;
    }
    .sl-empty-state h3 {
        margin-bottom: 0.4rem;
        font-weight: 650;
    }
    .sl-empty-state p {
        color: rgba(130, 130, 140, 0.95);
        font-size: 0.95rem;
        max-width: 480px;
        margin: 0 auto;
    }

    .sl-source-card {
        border: 1px solid rgba(140, 140, 160, 0.18);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        background: rgba(130, 130, 160, 0.04);
    }
    .sl-source-title {
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 0.15rem;
    }
    .sl-source-score {
        font-size: 0.76rem;
        color: rgba(130, 130, 140, 0.9);
        margin-bottom: 0.35rem;
    }
    .sl-source-excerpt {
        font-size: 0.83rem;
        color: rgba(110, 110, 120, 0.95);
        font-style: italic;
        line-height: 1.4;
    }

    .sl-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        background: rgba(60, 160, 110, 0.12);
        color: rgb(40, 140, 95);
        margin-left: 0.4rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "sources", "retrieval"}
    if "kb" not in st.session_state:
        st.session_state.kb = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []  # list of filenames already processed
    if "config_error" not in st.session_state:
        st.session_state.config_error = None


def get_knowledge_base() -> KnowledgeBase:
    """Lazily build the KnowledgeBase, loading the embedding model once."""
    if st.session_state.kb is None:
        config = get_config()
        model = get_embedding_model(config.embedding_model)
        st.session_state.kb = KnowledgeBase(
            embedding_model=model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            top_k=config.top_k,
            min_relevance_score=config.min_relevance_score,
        )
    return st.session_state.kb


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sl-brand">📚 StudyLens AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sl-brand-sub">Understand your documents. Ask anything.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sl-section-label">Knowledge Base</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="uploader",
        )

        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
            if new_files:
                process_uploads(new_files)

        if st.session_state.indexed_files:
            st.markdown('<div class="sl-section-label">Documents</div>', unsafe_allow_html=True)
            for name in st.session_state.indexed_files:
                st.markdown(f'<div class="sl-doc-chip">📄 {name}</div>', unsafe_allow_html=True)

        if st.session_state.kb and not st.session_state.kb.is_empty:
            stats = st.session_state.kb.stats()
            st.markdown('<div class="sl-section-label">Statistics</div>', unsafe_allow_html=True)
            for label, value in [
                ("Documents", stats.num_documents),
                ("Pages", stats.num_pages),
                ("Chunks", stats.num_chunks),
            ]:
                st.markdown(
                    f'<div class="sl-stat-row"><span class="sl-stat-label">{label}</span>'
                    f'<span class="sl-stat-value">{value:,}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sl-section-label">Settings</div>', unsafe_allow_html=True)
        config = get_config()
        st.caption(f"Model: `{config.openai_model}`")
        st.caption(f"Embedding: `{config.embedding_model}`")
        st.caption(f"Top-K retrieval: `{config.top_k}`")
        if config.min_relevance_score > 0:
            st.caption(f"Minimum relevance: `{config.min_relevance_score:.2f}`")
        if not config.openai_api_key:
            st.warning("OPENAI_API_KEY is not set. Add it to your .env file to enable answers.", icon="⚠️")

        st.markdown("---")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def process_uploads(files: list) -> None:
    """Ingest newly uploaded PDFs with a visible step-by-step progress state."""
    kb = get_knowledge_base()
    status_placeholder = st.sidebar.empty()

    for uploaded_file in files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            def progress(msg: str, _name=uploaded_file.name) -> None:
                status_placeholder.info(f"**{_name}**\n\n{msg}")

            kb.add_document(tmp_path, source_name=uploaded_file.name, progress=progress)
            st.session_state.indexed_files.append(uploaded_file.name)
            status_placeholder.success(f"✅ '{uploaded_file.name}' indexed.")

        except PDFLoadError as exc:
            status_placeholder.error(f"❌ {exc}")
        except (EmbeddingError, VectorStoreError) as exc:
            status_placeholder.error(f"❌ Failed to index '{uploaded_file.name}': {exc}")
        except Exception as exc:  # last-resort guard so the UI never crashes
            logger.exception("Unexpected error while indexing %s", uploaded_file.name)
            status_placeholder.error(f"❌ Unexpected error while indexing '{uploaded_file.name}'.")
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div class="sl-header">
            <h1>StudyLens AI</h1>
        </div>
        <div class="sl-tagline">Ask questions about your knowledge base.</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="sl-empty-state">
            <h3>📄 Upload your first document</h3>
            <p>StudyLens will read your documents and let you ask questions using natural
            language, with answers grounded in what they actually say.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for retrieved in sources:
            c = retrieved.chunk
            excerpt = c.text[:280] + ("…" if len(c.text) > 280 else "")
            st.markdown(
                f"""
                <div class="sl-source-card">
                    <div class="sl-source-title">📄 {c.source} — Page {c.page}</div>
                    <div class="sl-source-score">Relevance: {retrieved.score:.2f}</div>
                    <div class="sl-source-excerpt">"{excerpt}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_retrieval_details(retrieval_info) -> None:
    if retrieval_info is None:
        return
    query, chunks = retrieval_info
    with st.expander("🔍 Retrieval Details", expanded=False):
        st.markdown(f"**Query:** {query}")
        st.markdown(f"**Chunks retrieved:** {len(chunks)}")
        if not chunks:
            st.caption("No chunks met the retrieval threshold for this query.")
        for i, retrieved in enumerate(chunks, start=1):
            c = retrieved.chunk
            preview = c.text[:160] + ("…" if len(c.text) > 160 else "")
            st.markdown(
                f"**{i}. {c.source} — Page {c.page}**  \n"
                f"Similarity: `{retrieved.score:.3f}`  \n"
                f"> {preview}"
            )


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources"))
                render_retrieval_details(msg.get("retrieval"))


def handle_user_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    kb = st.session_state.kb
    config = get_config()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if kb is None or kb.is_empty:
            answer = "Please upload at least one PDF before asking a question."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        with st.spinner("Searching your documents..."):
            try:
                retrieval = kb.retriever.retrieve(question)
            except Exception as exc:
                logger.exception("Retrieval failed")
                st.error(f"Retrieval failed: {exc}")
                return

        with st.spinner("Generating answer..."):
            try:
                config.validate_for_llm()
                llm = LLMClient(api_key=config.openai_api_key, model=config.openai_model)
                history_for_prompt = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = llm.generate_answer(
                    question=question,
                    chunks=retrieval.chunks,
                    history=history_for_prompt,
                    max_history_turns=config.max_history_turns,
                )
                answer = result.answer
            except ConfigError as exc:
                answer = f"⚠️ {exc}"
                st.error(answer)
            except LLMError as exc:
                answer = f"⚠️ {exc}"
                st.error(answer)
            except Exception:
                logger.exception("Unexpected error during generation")
                answer = "⚠️ An unexpected error occurred while generating the answer."
                st.error(answer)
            else:
                st.markdown(answer)
                render_sources(retrieval.chunks)
                render_retrieval_details((retrieval.query, retrieval.chunks))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieval.chunks,
                        "retrieval": (retrieval.query, retrieval.chunks),
                    }
                )
                return

        # Only reached on an error path above.
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    init_session_state()
    render_sidebar()
    render_header()

    if st.session_state.kb is None or st.session_state.kb.is_empty:
        if not st.session_state.messages:
            render_empty_state()

    render_chat_history()

    question = st.chat_input("Ask a question about your documents...")
    if question:
        handle_user_question(question)


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
