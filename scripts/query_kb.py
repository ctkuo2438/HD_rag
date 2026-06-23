"""Smoke-test retrieval from an existing Phase 1 Chroma knowledge base."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from human_design.rag.config import load_config
from human_design.rag.retriever import create_retriever_from_config


DEFAULT_TOP_K = 3
DEFAULT_SNIPPET_CHARS = 500
REAL_EMBEDDINGS_ENV_VAR = "HD_RAG_REAL_EMBEDDINGS"
RETRIEVAL_DISABLED_MESSAGE = (
    "Real retrieval is disabled by default. "
    "Set HD_RAG_REAL_EMBEDDINGS=1 to run query retrieval."
)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run a retrieval smoke test against an existing Chroma collection.
    """
    args = _parse_args(argv)

    if os.environ.get(REAL_EMBEDDINGS_ENV_VAR) != "1":
        raise SystemExit(RETRIEVAL_DISABLED_MESSAGE)

    query = " ".join(args.query)

    try:
        config = load_config()
        retriever = create_retriever_from_config(
            config,
            similarity_top_k=args.top_k,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Could not load the existing knowledge base. Run or rebuild ingestion. {exc}"
        ) from exc

    results = retriever.retrieve(query)
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        _print_result(rank=rank, result=result, snippet_chars=args.snippet_chars)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k chunks from the existing Human Design Chroma store.",
    )
    parser.add_argument("query", nargs="+", help="Query text to retrieve against.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    return parser.parse_args(argv)


def _print_result(*, rank: int, result: Any, snippet_chars: int) -> None:
    node = getattr(result, "node", result)
    metadata = dict(getattr(node, "metadata", {}) or {})
    score = getattr(result, "score", None)
    snippet = _snippet(_node_text(node), snippet_chars)

    print(f"Result {rank}")
    if score is not None:
        print(f"Score: {score}")
    print(f"Snippet: {snippet}")
    _print_metadata("Source file", metadata.get("source_file"))
    _print_metadata("Source path", metadata.get("source_path"))
    _print_metadata("Page label", metadata.get("page_label"))
    _print_metadata("Page number", metadata.get("page_number"))


def _node_text(node: Any) -> str:
    get_content = getattr(node, "get_content", None)
    if get_content is not None:
        try:
            return str(get_content(metadata_mode="none"))
        except TypeError:
            return str(get_content())
    return str(getattr(node, "text", ""))


def _snippet(text: str, snippet_chars: int) -> str:
    snippet = text.strip()
    if len(snippet) <= snippet_chars:
        return snippet
    return f"{snippet[:snippet_chars]}..."


def _print_metadata(label: str, value: Any) -> None:
    if value is not None:
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
