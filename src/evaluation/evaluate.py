"""
Lightweight evaluation for the retrieval pipeline.

This is intentionally simple: it measures whether the retriever surfaces
the *expected* source document/page for a small hand-labeled set of
questions. It is meant to give directional signal during development, not
to be a rigorous benchmark. See README.md ("Evaluation") for methodology,
limitations, and how to interpret the numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.retrieval.retriever import Retriever


@dataclass
class EvalQuestion:
    """One labeled evaluation example."""

    question: str
    expected_source: str
    expected_page: Optional[int] = None


@dataclass
class EvalResult:
    """Per-question evaluation outcome."""

    question: str
    expected_source: str
    expected_page: Optional[int]
    hit: bool  # was the expected source (and page, if given) found in top-k?
    rank: Optional[int]  # 1-indexed rank of the first matching chunk, if hit
    top_k_sources: List[str]


@dataclass
class EvalSummary:
    """Aggregate metrics across all evaluation questions."""

    num_questions: int
    hit_rate: float  # fraction of questions where expected source appeared in top-k
    mean_reciprocal_rank: float  # average of 1/rank for hits, 0 for misses
    results: List[EvalResult]


def load_eval_dataset(path: str | Path) -> List[EvalQuestion]:
    """Load a JSON evaluation dataset of the form:

    [
      {"question": "...", "expected_source": "Operating Systems.pdf", "expected_page": 42},
      ...
    ]
    """
    data = json.loads(Path(path).read_text())
    return [
        EvalQuestion(
            question=item["question"],
            expected_source=item["expected_source"],
            expected_page=item.get("expected_page"),
        )
        for item in data
    ]


def evaluate_retrieval(
    retriever: Retriever,
    dataset: List[EvalQuestion],
    top_k: int | None = None,
) -> EvalSummary:
    """Run retrieval for every question in `dataset` and score hit rate / MRR.

    A "hit" means the expected source document (and expected page, if one
    was specified) appears among the retrieved top-k chunks for that
    question. This measures whether the retrieval stage is doing its job --
    it does not evaluate the quality of the final LLM-generated answer.
    """
    results: List[EvalResult] = []
    reciprocal_ranks: List[float] = []

    for item in dataset:
        retrieval = retriever.retrieve(item.question, top_k=top_k)
        rank: Optional[int] = None

        for i, retrieved in enumerate(retrieval.chunks, start=1):
            source_match = retrieved.chunk.source == item.expected_source
            page_match = item.expected_page is None or retrieved.chunk.page == item.expected_page
            if source_match and page_match:
                rank = i
                break

        hit = rank is not None
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)
        results.append(
            EvalResult(
                question=item.question,
                expected_source=item.expected_source,
                expected_page=item.expected_page,
                hit=hit,
                rank=rank,
                top_k_sources=[r.chunk.source for r in retrieval.chunks],
            )
        )

    n = len(dataset)
    hit_rate = sum(1 for r in results if r.hit) / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    return EvalSummary(
        num_questions=n,
        hit_rate=hit_rate,
        mean_reciprocal_rank=mrr,
        results=results,
    )


def summary_to_dict(summary: EvalSummary) -> dict:
    """Convert an EvalSummary to a plain dict (for JSON export or display)."""
    return {
        "num_questions": summary.num_questions,
        "hit_rate": round(summary.hit_rate, 4),
        "mean_reciprocal_rank": round(summary.mean_reciprocal_rank, 4),
        "results": [asdict(r) for r in summary.results],
    }
