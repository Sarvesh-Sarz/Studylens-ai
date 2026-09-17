"""
CLI to run the retrieval evaluation dataset against a set of real PDFs.

Usage:
    python run_evaluation.py --pdfs data/documents/*.pdf --eval data/eval/sample_questions.json

This indexes the given PDFs into a fresh in-memory KnowledgeBase, then runs
every question in the evaluation dataset against the retriever and prints
hit-rate / MRR metrics. It does NOT call the OpenAI API -- it only
evaluates the retrieval stage (see README.md "Evaluation" for why).

Note: `expected_source` / `expected_page` in the sample dataset are
illustrative placeholders. Replace them with values that match your own
PDFs before drawing conclusions from the reported metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from src.evaluation.evaluate import evaluate_retrieval, load_eval_dataset, summary_to_dict
from src.knowledge_base import KnowledgeBase
from src.retrieval.embeddings import get_embedding_model
from src.utils.config import get_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate StudyLens AI retrieval quality.")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Glob(s) of PDF files to index.")
    parser.add_argument("--eval", required=True, help="Path to the evaluation JSON dataset.")
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K for this run.")
    args = parser.parse_args()

    pdf_paths: list[str] = []
    for pattern in args.pdfs:
        pdf_paths.extend(glob.glob(pattern))
    pdf_paths = sorted(set(pdf_paths))

    if not pdf_paths:
        print("No PDF files matched the given --pdfs pattern(s).", file=sys.stderr)
        return 1

    config = get_config()
    print(f"Loading embedding model '{config.embedding_model}'...")
    model = get_embedding_model(config.embedding_model)

    kb = KnowledgeBase(
        embedding_model=model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=args.top_k or config.top_k,
    )

    for path in pdf_paths:
        name = Path(path).name
        print(f"Indexing {name}...")
        try:
            kb.add_document(path, source_name=name)
        except Exception as exc:
            print(f"  Skipped {name}: {exc}", file=sys.stderr)

    if kb.is_empty:
        print("No documents were successfully indexed. Aborting evaluation.", file=sys.stderr)
        return 1

    dataset = load_eval_dataset(args.eval)
    print(f"\nRunning {len(dataset)} evaluation questions...")
    summary = evaluate_retrieval(kb.retriever, dataset, top_k=args.top_k)

    print("\n=== Retrieval Evaluation Results ===")
    print(f"Questions evaluated : {summary.num_questions}")
    print(f"Hit rate            : {summary.hit_rate:.2%}")
    print(f"Mean reciprocal rank: {summary.mean_reciprocal_rank:.3f}")
    print()
    for r in summary.results:
        status = "HIT " if r.hit else "MISS"
        rank_str = f"(rank {r.rank})" if r.hit else ""
        print(f"[{status}] {r.question!r} -> expected {r.expected_source} p.{r.expected_page} {rank_str}")

    out_path = Path("data/eval/last_run_results.json")
    out_path.write_text(json.dumps(summary_to_dict(summary), indent=2))
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
