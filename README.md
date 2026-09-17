# StudyLens AI

**Understand your documents. Ask anything. Learn faster.**

A Retrieval-Augmented Generation (RAG) knowledge assistant for engineering students. Upload PDF textbooks or notes, ask questions in natural language, and get answers grounded in your own documents — with page-level citations.

---

## Overview

StudyLens AI lets a student upload one or more PDFs (e.g. Operating Systems, DBMS, Computer Networks lecture notes) and ask questions about them in a chat interface. Every answer is generated only from retrieved passages of the uploaded documents, and every answer shows exactly which document and page it came from. If the documents don't contain the answer, StudyLens says so instead of guessing.

## Problem

Engineering students juggle hundreds of pages of lecture notes and textbooks per subject. Finding a specific fact ("what are the four conditions for deadlock?") means manually searching PDFs, and generic chatbots either don't have the material or hallucinate plausible-sounding but wrong answers with no way to verify them.

## Solution

StudyLens indexes uploaded PDFs into a local semantic search index (FAISS) and only lets the LLM answer using the specific passages retrieved for that question. Every answer is traceable back to a document name, page number, and excerpt — turning an opaque chatbot into a transparent, verifiable study tool.

## Features

- Multi-PDF upload with live processing status (read → chunk → embed → index)
- Semantic search over all uploaded documents (FAISS + sentence-transformers)
- Grounded answer generation via the OpenAI API, with an explicit "not found in documents" fallback
- Per-answer **source cards**: document name, page number, relevance score, and excerpt
- **Retrieval Details** panel showing the raw retrieved chunks and similarity scores, for transparency
- Multi-document synthesis (e.g. "compare deadlock handling in OS with concurrency control in DBMS")
- Conversation memory for follow-up questions ("What are its four conditions?"), without letting prior chat turns count as document evidence
- Knowledge-base statistics (documents / pages / chunks), computed from the actual index
- Graceful error handling for missing keys, corrupted/scanned PDFs, rate limits, and empty queries

## Architecture

```
                          ┌─────────────────────┐
                          │     Streamlit UI     │
                          │       (app.py)        │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   KnowledgeBase       │
                          │ (src/knowledge_base)  │
                          └──┬────────┬────────┬──┘
                             │        │        │
              ┌──────────────▼──┐  ┌──▼─────┐ ┌▼──────────────┐
              │   Ingestion      │  │Retrieval│ │  Generation    │
              │ pdf_loader.py    │  │         │ │  llm.py        │
              │ chunker.py       │  │embeddings│ │  prompts.py    │
              │ (PyMuPDF)        │  │.py       │ │  (OpenAI API)  │
              │                  │  │vector_   │ │                │
              │                  │  │store.py  │ │                │
              │                  │  │(FAISS)   │ │                │
              │                  │  │retriever │ │                │
              │                  │  │.py       │ │                │
              └──────────────────┘  └─────────┘ └────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Evaluation          │
                          │ src/evaluation/       │
                          │ evaluate.py           │
                          └───────────────────────┘
```

Every box under `src/` is independently importable and independently testable — `app.py` never touches PyMuPDF, sentence-transformers, FAISS, or the OpenAI SDK directly. It only calls `KnowledgeBase` and `LLMClient`.

## RAG Pipeline

```
PDF file
   │  (pdf_loader.py — PyMuPDF)
   ▼
Per-page text  ──────────────────────────────┐
   │  (chunker.py — sliding window,          │  page number attached
   │   configurable size/overlap)            │  to every chunk
   ▼                                         │
Chunks + metadata {source, page, chunk_id}  ◄┘
   │  (embeddings.py — sentence-transformers)
   ▼
Embedding vectors (L2-normalized)
   │  (vector_store.py — FAISS IndexFlatIP)
   ▼
FAISS index  ◄── query embedding (same model) ── user question
   │  (retriever.py — top-k semantic search)
   ▼
Top-k relevant chunks (with similarity scores)
   │  (prompts.py — grounding instructions + chunks + history)
   ▼
LLM (llm.py — OpenAI chat completions)
   │
   ▼
Grounded answer + source citations (rendered by app.py)
```

## Tech Stack

| Layer            | Choice                              |
|-------------------|--------------------------------------|
| UI                | Streamlit                           |
| PDF extraction    | PyMuPDF (`fitz`)                    |
| Embeddings        | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store      | FAISS (`IndexFlatIP`, local, in-memory) |
| LLM               | OpenAI API (chat completions)       |
| Config            | `python-dotenv` + environment variables |
| Testing           | `pytest`                            |

## Project Structure

```
studylens-ai/
├── app.py                      # Streamlit UI (thin — delegates to src/)
├── run_evaluation.py           # CLI to run retrieval evaluation against real PDFs
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── pytest.ini
│
├── src/
│   ├── knowledge_base.py       # Orchestrates ingest → chunk → embed → index
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PyMuPDF text extraction, page-level
│   │   └── chunker.py          # Sliding-window chunking with metadata
│   ├── retrieval/
│   │   ├── embeddings.py       # sentence-transformers wrapper (swappable model)
│   │   ├── vector_store.py     # FAISS index + chunk/metadata storage
│   │   └── retriever.py        # Query embedding + top-k search
│   ├── generation/
│   │   ├── llm.py              # OpenAI API call + friendly error handling
│   │   └── prompts.py          # System prompt + grounded prompt construction
│   ├── evaluation/
│   │   └── evaluate.py         # Retrieval hit-rate / MRR scoring
│   └── utils/
│       └── config.py           # Environment-variable driven configuration
│
├── data/
│   ├── documents/               # (empty — user-uploaded PDFs are processed in-memory)
│   └── eval/
│       └── sample_questions.json
│
└── tests/
    ├── test_pdf_loader.py
    ├── test_chunker.py
    ├── test_config.py
    ├── test_vector_store.py
    ├── test_retriever.py
    ├── test_knowledge_base.py   # full-pipeline integration test
    ├── test_evaluate.py
    └── test_llm.py
```

## Setup

```bash
git clone <your-repo-url>
cd studylens-ai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Environment Variables

Set these in `.env` (see `.env.example`):

| Variable               | Purpose                                            | Default              |
|------------------------|-----------------------------------------------------|-----------------------|
| `OPENAI_API_KEY`       | Your OpenAI API key (required to generate answers)  | *(none — required)*  |
| `OPENAI_MODEL`         | Chat model used for generation                      | `gpt-4o-mini`         |
| `EMBEDDING_MODEL`      | sentence-transformers model name                    | `all-MiniLM-L6-v2`    |
| `CHUNK_SIZE`           | Target characters per chunk                         | `1000`                |
| `CHUNK_OVERLAP`        | Overlap between consecutive chunks (characters)     | `150`                 |
| `TOP_K`                | Number of chunks retrieved per question             | `5`                   |
| `MIN_RELEVANCE_SCORE`  | Optional similarity floor for keeping a chunk       | `0.0` (disabled)      |
| `MAX_HISTORY_TURNS`    | Recent conversation turns kept for follow-ups       | `6`                   |

You can upload PDFs and browse the knowledge-base UI without an API key; you'll only hit an error when you actually ask a question (clearly reported in the UI, not a stack trace).

## Running Locally

```bash
streamlit run app.py
```

Then open the printed local URL, upload one or more PDFs from the sidebar, wait for "Knowledge base ready," and start asking questions in the chat box.

## Example Usage

1. Upload `Operating Systems.pdf` and `Database Management Systems.pdf`.
2. Ask: *"What is deadlock?"* → StudyLens retrieves the relevant chunk(s), answers, and shows a source card like `📄 Operating Systems.pdf — Page 41`.
3. Ask a follow-up: *"What are its four necessary conditions?"* → the assistant uses conversation history to resolve "its" to deadlock, then retrieves fresh evidence for that specific sub-question (it does not simply reuse the previous answer as fact).
4. Ask: *"Compare deadlock handling in Operating Systems with concurrency control in DBMS."* → chunks are retrieved from both PDFs and cited separately.
5. Ask something not covered by your PDFs → StudyLens responds: *"I couldn't find enough information in the provided documents to answer that reliably."*

## Evaluation

### Dataset

`data/eval/sample_questions.json` is a small, hand-written **template** dataset (8 questions) in the format:

```json
{"question": "...", "expected_source": "Operating Systems.pdf", "expected_page": 42}
```

**Important honesty note:** the filenames and page numbers in the sample file are illustrative placeholders (no real "Operating Systems.pdf" was included in this repository, and no PDFs were indexed or evaluated in this environment). Before trusting any evaluation numbers, replace `expected_source`/`expected_page` with values that match the actual PDFs you index, or write your own dataset against your own documents.

### Methodology

Run:

```bash
python run_evaluation.py --pdfs "data/documents/*.pdf" --eval data/eval/sample_questions.json
```

This indexes the given PDFs, runs every question through the same `Retriever` the app uses, and checks whether the expected `(source, page)` appears among the top-k retrieved chunks for that question. It reports:

- **Hit rate** — fraction of questions where the expected source/page was retrieved in the top-k.
- **Mean reciprocal rank (MRR)** — rewards the expected chunk appearing higher in the ranking, not just anywhere in top-k.

This measures the **retrieval stage only** — it does not score whether the final LLM-generated answer is well-written or fully correct. A basic groundedness check (does the LLM's answer avoid claims unsupported by the retrieved context?) would require either manual review or an LLM-as-judge step, which is listed under Future Improvements rather than implemented here, to avoid presenting an unvalidated automatic score as a real metric.

### Limitations of this evaluation

- The sample dataset is small (8 questions) and template/placeholder data — a real evaluation needs a labeled set built against your actual PDFs.
- Retrieval hit-rate does not measure answer quality, tone, or completeness.
- Page-level expected answers assume the source concept is fully contained on a single page, which isn't always true for concepts that span a page break.

### Actual results

**No evaluation run was performed against real PDFs in this environment** (no textbook PDFs were available to index, and this build environment does not have OpenAI API access configured). The `run_evaluation.py` script is provided, tested for correct wiring against synthetic data (see `tests/test_evaluate.py`, which passes), and ready to run once you supply real PDFs — but no numbers are reported here because none were genuinely produced. Please run it yourself and report the real output.

## Design Decisions

**Why RAG instead of fine-tuning or a plain chatbot?** RAG lets the assistant answer from documents it was never trained on, keeps answers up to date with whatever the user uploads, and — critically for a study tool — makes every claim traceable to a citation instead of baked into opaque model weights.

**Why FAISS?** It's local, free, requires no external service or account, and `IndexFlatIP` gives exact (non-approximate) nearest-neighbor search, which is more than fast enough for the chunk counts a handful of uploaded PDFs produce. No need for a hosted vector database at this scale.

**Why semantic (dense) embeddings instead of keyword search?** Student questions rarely use the exact wording of the textbook ("why does a process get stuck forever" vs. "deadlock"). Dense embeddings retrieve by meaning, not just token overlap.

**Why page-level metadata?** Citations are only useful if a student can actually flip to the page and verify the claim. Chunking is done per-page specifically so every chunk can carry a single, correct page number — chunking the whole document as one blob first would make attributing a chunk to a page ambiguous.

**Why top-k retrieval instead of sending the whole document?** Sending entire PDFs to the LLM would blow past context limits for anything but the shortest documents, cost far more per query, and dilute the model's attention with irrelevant text. Retrieving a small, high-relevance set of chunks keeps the prompt focused and grounded.

**Why explicit grounding instructions in the system prompt?** LLMs are fluent enough to produce a plausible-sounding answer even with no supporting evidence. The system prompt explicitly forbids this and gives the model a safe, honest fallback sentence to use instead — the alternative (silent hallucination) is worse than an occasional "I don't know."

## Limitations

- **Scanned/image-only PDFs are not supported.** PyMuPDF only extracts text that exists as text in the PDF; scanned pages with no text layer would need OCR (not implemented).
- **Retrieval quality depends on chunking and embedding choices.** A small, general-purpose embedding model (`all-MiniLM-L6-v2`) is fast and free but not as accurate as larger, paid embedding APIs for subtle semantic distinctions.
- **API dependency.** Answer generation requires a working OpenAI API key and internet access; without it, only upload/browse/retrieval-inspection features work.
- **No persistent vector storage.** The FAISS index lives in memory for the Streamlit session; documents must be re-uploaded (and re-embedded) after a restart. There is no on-disk index persistence in this version.
- **Single-process, single-user design.** No authentication, no multi-user isolation — appropriate for a local study tool, not for a shared/hosted deployment as-is.
- **Chunking is character-based, not semantic.** It tries to break on natural boundaries (paragraphs/sentences) but doesn't understand document structure (headings, tables) the way a layout-aware chunker would.

## Future Improvements

- Hybrid search (dense + keyword/BM25) to catch exact-term queries embeddings sometimes miss
- Cross-encoder reranking of retrieved chunks before generation
- OCR fallback (e.g. Tesseract) for scanned PDFs
- Persistent FAISS index on disk (or a hosted vector DB) so uploads survive restarts
- LLM-as-judge groundedness scoring as a second evaluation axis, clearly labeled as an approximate signal
- Streaming generation for faster perceived response time
- Basic user accounts if the app moves beyond a single local session

## AI Tools Disclosure

This project was built with the assistance of an AI coding assistant (Claude, by Anthropic), which was used to design the module structure, write the ingestion/retrieval/generation pipeline, the Streamlit UI, the test suite, and this README. It was not written entirely manually. All code was reviewed for correctness and internal consistency, and the automated tests in `tests/` were actually executed (42/42 passing at the time of writing, including a full-pipeline integration test in `tests/test_knowledge_base.py` that exercises ingest → chunk → embed → index → retrieve end to end on a real generated PDF) rather than assumed to pass. No evaluation metrics, screenshots, or benchmark numbers were fabricated — the "Actual results" section above states plainly that no live evaluation run was performed, since no real textbook PDFs or OpenAI credentials were available in the build environment.

One specific thing worth disclosing: the real `sentence-transformers` embedding model could not be downloaded in the build sandbox (its network is restricted to package registries and cannot reach `huggingface.co`), so the retrieval pipeline was tested with a small deterministic fake embedding model instead of the real one. That test run did usefully confirm one real thing: attempting to load the real model failed gracefully with the intended `EmbeddingError` message rather than crashing with a raw traceback, which is the behavior `embeddings.py` was designed to produce. The real model will download normally the first time you run the app with internet access.

## What Should Be Tested Manually

Before an interview or demo, manually verify:

1. Uploading a real multi-page PDF and confirming citations point to the correct pages.
2. Asking a question with no answer in the documents and confirming the "not found" fallback triggers instead of a hallucinated answer.
3. Asking a follow-up question ("What about its causes?") and confirming it's understood in context.
4. Uploading two PDFs and asking a comparison question that should cite both.
5. Removing/blanking `OPENAI_API_KEY` and confirming the UI shows a friendly warning rather than crashing.
6. Uploading a corrupted or non-PDF file and confirming a graceful error message.
7. Running `pytest` and confirming all tests pass in your environment.
