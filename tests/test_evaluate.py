"""Tests for src.evaluation.evaluate, using the same fake embedding model
approach as test_retriever.py to avoid downloading real model weights.
"""

from __future__ import annotations

import json

from src.evaluation.evaluate import EvalQuestion, evaluate_retrieval, load_eval_dataset
from src.retrieval.retriever import Retriever
from tests.test_retriever import FakeEmbeddingModel, _build_store


def test_evaluate_retrieval_scores_hits_and_misses():
    model, store = _build_store()
    retriever = Retriever(model, store, top_k=2)

    dataset = [
        EvalQuestion(question="What is a deadlock?", expected_source="OS.pdf", expected_page=41),
        EvalQuestion(question="What is networking?", expected_source="WRONG.pdf", expected_page=None),
    ]

    summary = evaluate_retrieval(retriever, dataset)

    assert summary.num_questions == 2
    assert summary.results[0].hit is True
    assert summary.results[0].rank == 1
    assert summary.results[1].hit is False
    assert 0.0 <= summary.hit_rate <= 1.0
    assert 0.0 <= summary.mean_reciprocal_rank <= 1.0


def test_evaluate_retrieval_on_empty_dataset_returns_zeroed_summary():
    model, store = _build_store()
    retriever = Retriever(model, store, top_k=2)

    summary = evaluate_retrieval(retriever, [])

    assert summary.num_questions == 0
    assert summary.hit_rate == 0.0
    assert summary.mean_reciprocal_rank == 0.0


def test_load_eval_dataset_parses_json_file(tmp_path):
    data = [
        {"question": "Q1", "expected_source": "A.pdf", "expected_page": 1},
        {"question": "Q2", "expected_source": "B.pdf"},
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data))

    questions = load_eval_dataset(path)

    assert len(questions) == 2
    assert questions[0].expected_page == 1
    assert questions[1].expected_page is None
