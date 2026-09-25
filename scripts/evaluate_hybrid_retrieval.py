#!/usr/bin/env python
"""Compare retrieval modes over an existing Phase 3 index; never build vectors."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from human_design.rag.bm25 import load_bm25_retriever
from human_design.rag.config import AppConfig, load_config
from human_design.rag.evaluation import (
    REGRESSION_THRESHOLDS,
    RetrievalMetrics,
    evaluate_modes,
    load_evaluation_cases,
    threshold_failures,
)
from human_design.rag.hybrid_index import require_real_embeddings
from human_design.rag.hybrid_retriever import Retriever
from human_design.rag.retriever import load_dense_retriever


def _format_metrics(metrics: RetrievalMetrics) -> str:
    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"
    return (f"Hit@5={rate(metrics.hit_at_5)} MRR@20={rate(metrics.mrr_at_20)} "
            f"entity={rate(metrics.exact_entity_hit_rate)} source/page={rate(metrics.expected_source_page_hit_rate)}")


def main(
    argv: Sequence[str] | None = None, *,
    retrievers: tuple[Retriever, Retriever] | None = None, config: AppConfig | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense-only, BM25-only, and RRF hybrid retrieval.",
        epilog="Exit codes: 0 thresholds passed; 1 aggregate regression thresholds failed; "
               "2 invalid input/config/index setup. Unlabeled metrics are excluded from thresholds.",
    )
    parser.add_argument("--queries", type=Path, required=True, help="Sanitized query labels JSON.")
    args = parser.parse_args(argv)
    try:
        cases = load_evaluation_cases(args.queries)
        settings = config if config is not None else load_config()
        if retrievers is None:
            require_real_embeddings(settings)
            dense = load_dense_retriever(settings)
            sparse = load_bm25_retriever(settings.index_dir / "bm25", dense.ingestion_id)
        else:
            dense, sparse = retrievers
        reports = evaluate_modes(cases, dense, sparse, settings)
    except ValueError as exc:
        print(f"Cannot evaluate retrieval: {exc}", file=sys.stderr)
        return 2
    failed = False
    for mode, report in reports.items():
        print(mode)
        for row in report.per_query:
            print(f"  {row.case_id} ({row.query}): {_format_metrics(row.metrics)}")
        print(f"  aggregate: {_format_metrics(report.aggregate)}")
        for name in threshold_failures(report.aggregate):
            failed = True
            print(f"  {mode}: threshold failure: {name}={getattr(report.aggregate, name):.4f} "
                  f"< {REGRESSION_THRESHOLDS[name]:.4f}")
    print("Regression thresholds FAILED" if failed else "Regression thresholds passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
