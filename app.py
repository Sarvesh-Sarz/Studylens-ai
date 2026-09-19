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
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header { visibility: hidden; }

    :root {
        --sl-bg: #f7f7f5;
        --sl-surface: #ffffff;
        --sl-border: #e4e4e1;
        --sl-border-strong: #d4d4d0;
        --sl-text: #171717;
        --sl-muted: #737373;
        --sl-soft: #f1f1ef;
    }

    .stApp {
        background: var(--sl-bg);
        color: var(--sl-text);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 3.25rem;
        padding-bottom: 6rem;
    }

    [data-testid="stSidebar"] {
        background: #f1f1ef;
        border-right: 1px solid var(--sl-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.15rem;
    }

    [data-testid="stFileUploader"] section {
        border: 1px solid var(--sl-border-strong) !important;
        background: var(--sl-surface) !important;
        padding: 0.75rem !important;
    }

    [data-testid="stFileUploader"] section > div {
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] button {
        border: 1px solid #222 !important;
        background: #222 !important;
        color: white !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: var(--sl-muted) !important;
    }

    .sl-brand {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--sl-text);
        margin-bottom: 0.2rem;
    }

    .sl-brand-sub {
        font-size: 0.78rem;
        line-height: 1.45;
        color: var(--sl-muted);
        margin-bottom: 1.65rem;
    }

    .sl-section-label {
        text-transform: uppercase;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        color: #8a8a86;
        margin-top: 1.35rem;
        margin-bottom: 0.55rem;
    }

    .sl-doc-chip {
        font-size: 0.78rem;
        line-height: 1.35;
        padding: 0.6rem 0.7rem;
        border: 1px solid var(--sl-border); 
        background: var(--sl-surface);
        margin-bottom: 0.4rem;
        overflow-wrap: anywhere;
        color: #303030;
    }

    .sl-stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        padding: 0.22rem 0;
    }

    .sl-stat-label { color: var(--sl-muted); }
    .sl-stat-value { font-weight: 650; color: var(--sl-text); }

    .sl-setting {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.34rem 0;
        font-size: 0.75rem;
        border-bottom: 1px solid #e9e9e6;
    }

    .sl-setting:last-child { border-bottom: 0; }
    .sl-setting-label { color: var(--sl-muted); }
    .sl-setting-value { color: #333; text-align: right; }

    .sl-header {
        padding: 0.5rem 0 1.2rem;
        border-bottom: 1px solid var(--sl-border);
        margin-bottom: 1.6rem;
    }

    .sl-header h1 {
        font-size: 1.75rem;
        line-height: 1.15;
        font-weight: 720;
        letter-spacing: -0.045em;
        margin: 0;
        color: var(--sl-text);
    }

    .sl-tagline {
        color: var(--sl-muted);
        font-size: 0.88rem;
        margin-top: 0.45rem;
    }

    .sl-empty-state {
        max-width: 650px;
        margin: 4.5rem auto 2rem;
        text-align: center;
        padding: 2.5rem 1.5rem;
        border: 1px solid var(--sl-border);
        background: rgba(255,255,255,0.58);
    }

    .sl-empty-state h3 {
        font-size: 1.05rem;
        font-weight: 680;
        letter-spacing: -0.02em;
        margin: 0 0 0.55rem;
    }

    .sl-empty-state p {
        color: var(--sl-muted);
        font-size: 0.88rem;
        line-height: 1.6;
        max-width: 500px;
        margin: 0 auto;
    }

    .sl-message {
        max-width: 820px;
        margin: 0 auto 1.35rem;
    }

    .sl-message-user {
        margin-left: auto;
        max-width: 78%;
        background: #e9e9e7;
        border: 1px solid #dededb;
        padding: 0.85rem 1rem;
        color: #222;
        line-height: 1.55;
        font-size: 0.91rem;
    }

    .sl-message-assistant {
        padding: 0.15rem 0 0.15rem;
        color: #202020;
        line-height: 1.7;
        font-size: 0.93rem;
    }

    .sl-message-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #858580;
        margin-bottom: 0.35rem;
    }

    .sl-source-card {
        border: 1px solid var(--sl-border);
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.5rem;
        background: #fbfbfa;
    }

    .sl-source-title {
        font-weight: 650;
        font-size: 0.82rem;
        margin-bottom: 0.18rem;
    }

    .sl-source-score {
        font-size: 0.7rem;
        color: #8a8a86;
        margin-bottom: 0.35rem;
    }

    .sl-source-excerpt {
        font-size: 0.77rem;
        color: #6c6c68;
        line-height: 1.5;
    }

    .sl-divider {
        height: 1px;
        background: var(--sl-border);
        margin: 1.4rem 0;
    }

    .sl-upload-note {
        font-size: 0.72rem;
        color: #858580;
        margin-top: 0.45rem;
    }

    .sl-footer-note {
        text-align: center;
        color: #999993;
        font-size: 0.7rem;
        margin-top: 1.25rem;
    }

    /* Native Streamlit controls */
    .stButton > button {
        border: 1px solid #d6d6d2 !important;
        background: #fff !important;
        color: #222 !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        border-color: #999 !important;
        background: #f8f8f6 !important;
    }

    [data-testid="stChatInput"] {
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid #cfcfcb !important;
        background: #fff !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.04) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #171717 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stChatInput"] button {
        color: #222 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--sl-border) !important;
        background: #fff !important;
    }

    [data-testid="stAlert"] {
    }

    code {
        color: #333 !important;
        background: #ededeb !important;
    }

    /* Remove Streamlit's default rounded corners */

    [data-testid="stChatInput"] *,
    [data-testid="stFileUploader"] *,
    [data-testid="stExpander"] *,
    .stButton > button {
        border-radius: 0 !important;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 0 !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 0 !important;
    }

    [data-testid="stExpander"] {
        border-radius: 0 !important;
    }

    [data-testid="stExpander"] details {
        border-radius: 0 !important;
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
        st.markdown('<div class="sl-brand">StudyLens AI</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="sl-doc-chip">{name}</div>', unsafe_allow_html=True)

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
        st.caption(f"Model: `{config.gemini_model}`")
        st.caption(f"Embedding: `{config.embedding_model}`")
        st.caption(f"Top-K retrieval: `{config.top_k}`")
        if config.min_relevance_score > 0:
            st.markdown(
                f'<div class="sl-setting"><span class="sl-setting-label">Min relevance</span><span class="sl-setting-value">{config.min_relevance_score:.2f}</span></div>',
                unsafe_allow_html=True,
            )
        if not config.gemini_api_key:
            st.warning("GEMINI_API_KEY is not set. Add it to your .env file to enable answers.")

        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
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
            <h3>Upload a document to get started</h3>
            <p>StudyLens will read your documents and let you ask questions using natural
            language, with answers grounded in what they actually say.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
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
    with st.expander("Retrieval details", expanded=False):
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
        role = msg["role"]
        if role == "user":
            st.markdown(
                f'<div class="sl-message"><div class="sl-message-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sl-message"><div class="sl-message-label">StudyLens</div></div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
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

    if kb is None or kb.is_empty:
        answer = "Please upload at least one PDF before asking a question."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return

    with st.status("Searching your documents...", expanded=False) as status:
        try:
            retrieval = kb.retriever.retrieve(question)
            status.update(label="Documents retrieved", state="complete")
        except Exception as exc:
            logger.exception("Retrieval failed")
            answer = f"Retrieval failed: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": answer})
            status.update(label="Retrieval failed", state="error")
            return

    try:
        config.validate_for_llm()
        llm = LLMClient(api_key=config.gemini_api_key, model=config.gemini_model)
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
        answer = f"Configuration error: {exc}"
    except LLMError as exc:
        answer = str(exc)
    except Exception:
        logger.exception("Unexpected error during generation")
        answer = "An unexpected error occurred while generating the answer. Please try again."

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieval.chunks,
            "retrieval": (retrieval.query, retrieval.chunks),
        }
    )
    st.rerun()


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
