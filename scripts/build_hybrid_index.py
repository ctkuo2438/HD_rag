"""Build a fresh local Phase 3 hybrid index with explicit embedding-cost opt-in."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from human_design.rag.config import load_config
from human_design.rag.hybrid_index import build_hybrid_index, require_real_embeddings


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a fresh Phase 3 index; real embedding cost may occur.")
    parser.parse_args(argv)
    try:
        config = load_config()
        require_real_embeddings(config)
        print("Real embeddings are enabled; OpenAI embedding cost may occur.")
        manifest = build_hybrid_index(config)
    except ValueError as exc:
        raise SystemExit(f"Could not build Phase 3 hybrid index: {exc}") from None
    print(f"Hybrid index verified: sources={len(manifest.source_fingerprints)} chunks={manifest.chunk_count}")
    print(f"ingestion_id={manifest.ingestion_id}")


if __name__ == "__main__":
    main()
