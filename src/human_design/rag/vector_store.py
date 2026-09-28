"""Persistent Chroma client for the hybrid index's explicit storage directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


def create_chroma_client(chroma_dir: Path) -> Any:
    """Open a local client at the directory selected by the hybrid index."""
    return chromadb.PersistentClient(path=str(chroma_dir))
