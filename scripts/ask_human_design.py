#!/usr/bin/env python
"""Ask one focused Human Design question using the opt-in Phase 3 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from human_design.rag.hybrid_index import HybridIndexError
from human_design.reading import ChartImageQuestionRequest, KnowledgeQuestionRequest, ReadingPipeline
from human_design.reading.models import AnswerResult
from human_design.reading.pipeline import ReadingPipelineError
from human_design.reading.query_builder import InvalidQuestionError


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default error includes raw argument values and local paths.
        self.exit(2, "Invalid arguments. Use --help for the supported interface.\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="ask_human_design.py", allow_abbrev=False,
        description="Ask one focused Human Design question; full readings are deferred.",
    )
    parser.add_argument("query", help="One focused question (at most 2,000 Python characters).")
    parser.add_argument("--bodygraph", type=Path, help="Local chart image for the opt-in Phase 2 extraction facade.")
    parser.add_argument("--json", action="store_true", help="Print one public AnswerResult JSON object.")
    return parser


def _print_answer(result: AnswerResult) -> None:
    print(f"Status: {result.status.value}")
    print(result.answer_markdown)
    for citation in result.citations:
        page = citation.page_label if citation.page_label is not None else citation.page_number
        location = f", page {page}" if page is not None else ""
        print(f"[{citation.citation_id}] {citation.source_file}{location}")
    for warning in result.warnings:
        print(f"Warning: {warning}")


def main(argv: Sequence[str] | None = None, *, pipeline: ReadingPipeline | None = None) -> int:
    args = _build_parser().parse_args(argv)
    service = pipeline if pipeline is not None else ReadingPipeline()
    try:
        if args.bodygraph is None:
            result = service.answer_knowledge_question(KnowledgeQuestionRequest(args.query))
        else:
            result = service.answer_chart_image_question(ChartImageQuestionRequest(args.query, args.bodygraph))
    except (InvalidQuestionError, HybridIndexError, ReadingPipelineError) as exc:
        # These core errors contain project-owned safe messages, never provider payloads.
        print(f"Cannot answer: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("Reading failed. Check local input and provider configuration.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, separators=(",", ":")))
    else:
        _print_answer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
