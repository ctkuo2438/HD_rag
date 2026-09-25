"""Offline sparse retrieval over canonical nodes, with a project-owned identity sidecar."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.retrievers.bm25 import BM25Retriever

from human_design.rag.hybrid_index import (
    HybridIndexError,
    canonical_json,
    require_empty_directory,
    safe_source_metadata,
    validate_canonical_nodes,
)
from human_design.rag.models import RetrievedChunk
from human_design.vision.constants import CENTER_ALIASES


# Include single-digit Gate/Line/Profile terms in the otherwise ordinary English tokenizer.
_TOKEN_PATTERN = r"(?u)\b\w+\b"
_G_CENTER = CENTER_ALIASES["G Center"]
_EGO_CENTER = CENTER_ALIASES["Heart"]
_ALIASES = (
    ("Manifesting Generator", ("顯示生產者", "显示生产者")),
    ("Generator", ("生產者", "生产者")),
    ("Manifestor", ("顯示者", "显示者")),
    ("Projector", ("投射者",)), ("Reflector", ("反映者",)),
    ("Sacral Authority", ("薦骨權威", "荐骨权威")),
    ("Emotional Authority", ("情緒權威", "情绪权威", "Solar Plexus Authority")),
    ("Solar Plexus Authority", ("情緒權威", "情绪权威", "Emotional Authority")),
    ("Splenic Authority", ("脾臟權威", "脾脏权威", "直覺權威", "直觉权威")),
    ("Ego Authority", ("意志力權威", "意志力权威")),
    ("Self-Projected Authority", ("自我投射權威", "自我投射权威")),
    ("Mental Authority", ("心智權威", "心智权威")),
    ("Lunar Authority", ("月亮權威", "月亮权威")),
    ("Head", ("頭頂中心", "头顶中心", "Head Center")),
    ("Ajna", ("邏輯中心", "逻辑中心", "Ajna Center")),
    ("Throat", ("喉嚨中心", "喉咙中心", "Throat Center")),
    (_G_CENTER, tuple(alias for alias, center in CENTER_ALIASES.items() if center == _G_CENTER)
     + ("G中心", "自我中心")),
    (_EGO_CENTER, tuple(alias for alias, center in CENTER_ALIASES.items() if center == _EGO_CENTER)
     + ("意志力中心", "心臟中心", "心脏中心", "Heart Center", "Ego Center")),
    ("Sacral", ("薦骨中心", "荐骨中心", "Sacral Center")),
    ("Solar Plexus", ("情緒中心", "情绪中心", "Solar Plexus Center")),
    ("Spleen", ("脾臟中心", "脾脏中心", "Spleen Center")),
    ("Root", ("根部中心", "Root Center")),
    ("Personality", ("人格", "意識", "意识")), ("Design", ("設計", "设计")),
    ("Sun", ("太陽", "太阳")), ("Earth", ("地球",)), ("Moon", ("月亮",)),
    ("North Node", ("北交點", "北交点")), ("South Node", ("南交點", "南交点")),
    ("Mercury", ("水星",)), ("Venus", ("金星",)), ("Mars", ("火星",)),
    ("Jupiter", ("木星",)), ("Saturn", ("土星",)), ("Uranus", ("天王星",)),
    ("Neptune", ("海王星",)), ("Pluto", ("冥王星",)), ("Line", ("爻",)),
)
_GATE = re.compile(
    r"\b(?:Gate|Hexagram)\s*#?\s*(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\s+Gate\b"
    r"|(\d{1,2})\s*(?:號|号)?\s*(?:閘門|闸门)", re.I,
)
_PROFILE = re.compile(
    r"\bProfile\s*([1-6])\s*[/\-]\s*([1-6])(?!\d)"
    r"|(?<!\d)([1-6])\s*[/\-]\s*([1-6])\s*(?:Profile\b|人生角色)", re.I,
)
_CHANNEL = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:[-–—]|(?:號|号)?\s*(?:與|与))\s*(\d{1,2})(?!\d)",
)


def _contains(text: str, phrase: str) -> bool:
    # G in G中心 (or Ego in ego_center) is not a standalone canonical token.
    if phrase in {_G_CENTER, _EGO_CENTER}:
        return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I) is not None
    # Preserve the distinct authority phrase when recognizing the standalone Self alias.
    if phrase == "Self":
        return re.search(r"(?<![\w-])Self(?![\w-])", text, re.I) is not None
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", text, re.I) is not None


def normalize_sparse_query(query: str) -> str:
    """Preserve useful words and append a deterministic, idempotent set of exact phrases."""
    result = " ".join(query.split())
    phrases: list[str] = []
    for match in _GATE.finditer(result):
        gate = int(next(group for group in match.groups() if group is not None))
        if 1 <= gate <= 64:
            phrases.extend((f"Gate {gate}", f"Hexagram {gate}"))
    profiles = list(_PROFILE.finditer(result))
    for match in profiles:
        first, second = (group for group in match.groups() if group is not None)
        phrases.extend((f"Profile {first}/{second}", f"Profile {first} {second}"))
    for match in _CHANNEL.finditer(result):
        if any(profile.start() <= match.start() < profile.end() for profile in profiles):
            continue
        first, second = sorted(map(int, match.groups()))
        if 1 <= first <= 64 and 1 <= second <= 64 and first != second:
            phrases.extend((f"Channel {first}-{second}", f"Channel {second}-{first}"))
    for canonical, aliases in _ALIASES:
        if any(_contains(result, alias) for alias in (canonical, *aliases)):
            phrases.append(canonical)
    for phrase in phrases:
        if not _contains(result, phrase):
            result = f"{result} {phrase}".strip()
    return result


def _corpus_nodes(retriever: BM25Retriever) -> list[BaseNode]:
    nodes = [metadata_dict_to_node(row) for row in retriever.corpus]
    validate_canonical_nodes(nodes)
    if any(row["node_id"] != node.node_id for row, node in zip(retriever.corpus, nodes, strict=True)):
        raise HybridIndexError("BM25 corpus node identity mismatch")
    return nodes


def build_and_persist_bm25(nodes: list[BaseNode], persist_dir: Path, ingestion_id: str) -> None:
    validate_canonical_nodes(nodes)
    require_empty_directory(persist_dir)
    if not nodes or not isinstance(ingestion_id, str) or not ingestion_id.strip():
        raise HybridIndexError("BM25 requires canonical nodes and an ingestion identity")
    for node in nodes:
        if node.metadata.get("ingestion_id", ingestion_id) != ingestion_id:
            raise HybridIndexError("BM25 node ingestion identity mismatch")
        node.metadata["ingestion_id"] = ingestion_id
        if "ingestion_id" not in node.excluded_embed_metadata_keys:
            node.excluded_embed_metadata_keys.append("ingestion_id")
        if "ingestion_id" not in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.append("ingestion_id")
    try:
        retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=min(2, len(nodes)), token_pattern=_TOKEN_PATTERN, verbose=False,
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        retriever.persist(str(persist_dir), show_progress=False)
        identity = {"ingestion_id": ingestion_id, "chunk_count": len(nodes),
                    "chunk_ids": [node.node_id for node in nodes], "token_pattern": _TOKEN_PATTERN}
        with (persist_dir / "identity.json").open("x", encoding="utf-8") as output:
            output.write(canonical_json(identity) + "\n")
    except Exception:
        raise HybridIndexError("Cannot build or persist BM25 index") from None


def load_bm25_retriever(persist_dir: Path, expected_ingestion_id: str) -> BM25RetrieverAdapter:
    """Read only the BM25 directory. The caller owns root-manifest validation."""
    try:
        identity = json.loads((persist_dir / "identity.json").read_text(encoding="utf-8"))
        if identity["ingestion_id"] != expected_ingestion_id:
            raise HybridIndexError("BM25 ingestion identity mismatch")
        if type(identity["chunk_count"]) is not int or identity["chunk_count"] <= 0:
            raise HybridIndexError("BM25 chunk count is invalid")
        if identity["token_pattern"] != _TOKEN_PATTERN:
            raise HybridIndexError("BM25 tokenization identity mismatch")
        retriever = BM25Retriever.from_persist_dir(str(persist_dir), show_progress=False)
        # 0.8.0 persists Top-K/verbosity/mask, but not token_pattern. Restore our
        # explicitly recorded setting so single-digit terms survive query reload.
        retriever.token_pattern = identity["token_pattern"]
        nodes = _corpus_nodes(retriever)
        if len(nodes) != identity["chunk_count"] or [node.node_id for node in nodes] != identity["chunk_ids"]:
            raise HybridIndexError("BM25 corpus count or ordered identity mismatch")
        if any(node.metadata.get("ingestion_id") != expected_ingestion_id for node in nodes):
            raise HybridIndexError("BM25 corpus ingestion identity mismatch")
        if int(retriever.bm25.scores["num_docs"]) != len(nodes):
            raise HybridIndexError("BM25 indexed document count mismatch")
        return BM25RetrieverAdapter(retriever, expected_ingestion_id)
    except HybridIndexError:
        raise
    except Exception:
        raise HybridIndexError("Cannot load missing or malformed BM25 index/identity") from None


@dataclass(frozen=True)
class BM25RetrieverAdapter:
    _retriever: BM25Retriever = field(repr=False)
    ingestion_id: str

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in _corpus_nodes(self._retriever))

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        normalized = normalize_sparse_query(query)
        if not normalized:
            return []
        previous = self._retriever.similarity_top_k
        try:
            self._retriever.similarity_top_k = min(top_k, len(self._retriever.corpus))
            matches = self._retriever.retrieve(normalized)
        finally:
            self._retriever.similarity_top_k = previous
        results = []
        for rank, match in enumerate(matches, 1):
            metadata = safe_source_metadata(match.node.metadata)
            if metadata.get("chunk_id") != match.node.node_id or "source_file" not in metadata:
                raise HybridIndexError("BM25 result lacks canonical source identity")
            results.append(RetrievedChunk(
                chunk_id=match.node.node_id, text=match.node.get_content(metadata_mode=MetadataMode.NONE),
                source_file=metadata["source_file"], source_relpath=metadata.get("source_relpath"),
                document_title=metadata.get("document_title"), page_label=metadata.get("page_label"),
                page_number=metadata.get("page_number"), sparse_rank=rank, sparse_score=match.score, metadata=metadata,
            ))
        return results
