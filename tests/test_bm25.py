"""Temporary, offline BM25 persistence and exact-term normalization tests."""

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

from human_design.rag import bm25
from human_design.rag import hybrid_index as hybrid
from human_design.vision.constants import CANONICAL_CENTERS, CENTER_ALIASES


def _nodes(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "synthetic.pdf").write_bytes(b"synthetic source")
    chunks = [TextNode(text=text, metadata={
        "source_file": "synthetic.pdf", "source_path": str(pdf_dir / "synthetic.pdf"),
        "page_label": str(i), "page_number": i, "document_title": "Synthetic reference",
        "private_cache_path": str(tmp_path / "private-cache"),
    }) for i, text in enumerate((
        "Gate 42 completes cycles. Hexagram 42.",
        "Gate 1 expresses creativity. Hexagram 1.",
        "Sacral Authority responds. Profile 4/6 is a profile.",
    ), 1)]
    return hybrid.create_canonical_nodes(chunks, hybrid.fingerprint_sources(pdf_dir), pdf_dir)


@pytest.mark.parametrize("query", ["Gate 42", " gate   42 ", "42nd Gate", "Hexagram 42", "42號閘門", "42閘門"])
def test_gate_normalization(query: str) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert "gate 42" in normalized.casefold()
    assert "hexagram 42" in normalized.casefold()
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize("query", ["Channel 42-53", "Channel 53-42", "42 - 53", "42號與53號通道"])
def test_channel_normalization(query: str) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert "channel 42-53" in normalized.casefold()
    assert "channel 53-42" in normalized.casefold()
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize("query", ["Profile 4/6", "4-6 Profile", "4/6人生角色"])
def test_profile_normalization(query: str) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert "profile 4/6" in normalized.casefold()
    assert "profile 4 6" in normalized.casefold()
    assert "Channel" not in normalized
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize(("query", "terms"), [
    ("薦骨權威", ("Sacral Authority",)), ("情緒權威", ("Emotional Authority", "Solar Plexus Authority")),
    ("脾臟權威", ("Splenic Authority",)), ("emotional authority", ("Solar Plexus Authority",)),
    ("Solar Plexus Authority", ("Emotional Authority",)),
    ("生產者 投射者 顯示者 反映者 顯示生產者", ("Generator", "Projector", "Manifestor", "Reflector", "Manifesting Generator")),
    ("頭頂中心 喉嚨中心 根部中心", ("Head", "Throat", "Root")),
    ("人格太陽 設計地球 月亮 北交點 南交點", ("Personality", "Sun", "Design", "Earth", "Moon", "North Node", "South Node")),
    ("木星 土星 水星 金星 火星 天王星 海王星 冥王星", ("Jupiter", "Saturn", "Mercury", "Venus", "Mars", "Uranus", "Neptune", "Pluto")),
])
def test_explicit_aliases(query: str, terms: tuple[str, ...]) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert all(term.casefold() in normalized.casefold() for term in terms)
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize(("query", "expected"), [
    ("G", "G"),
    ("G Center", "G Center"),
    ("Self", "Self G"),
    ("Identity", "Identity G"),
    (" g   cEnTeR ", "g cEnTeR"),
    ("G中心", "G中心 G"),
    ("我的G中心", "我的G中心 G"),
    ("自我中心", "自我中心 G"),
    ("Heart", "Heart Ego"),
    ("Will", "Will Ego"),
    ("Ego", "Ego"),
    ("Heart Center", "Heart Center Ego"),
    ("Ego Center", "Ego Center"),
    (" wIlL   center ", "wIlL center Ego"),
    ("意志力中心", "意志力中心 Ego"),
    ("心臟中心", "心臟中心 Ego"),
    ("心脏中心", "心脏中心 Ego"),
])
def test_g_and_ego_queries_preserve_aliases_and_append_only_canonical_terms(
    query: str, expected: str,
) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert normalized == expected
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize(("alias", "canonical"), [
    (alias, canonical) for alias, canonical in CENTER_ALIASES.items()
    if canonical in {"G", "Ego"}
])
def test_sparse_g_and_ego_aliases_match_phase2_vocabulary(alias: str, canonical: str) -> None:
    assert canonical in CANONICAL_CENTERS
    normalized = bm25.normalize_sparse_query(alias)
    # Unicode word boundaries prevent G in G中心 or g_center counting as a standalone term.
    assert re.search(r"(?<!\w)" + re.escape(canonical) + r"(?!\w)", normalized, re.I)
    assert bm25.normalize_sparse_query(normalized) == normalized


def test_g_center_aliases_emit_one_standalone_canonical_term() -> None:
    normalized = bm25.normalize_sparse_query("G中心 自我中心 Self Identity")
    assert normalized == "G中心 自我中心 Self Identity G"
    assert len(re.findall(r"(?<!\w)G(?!\w)", normalized)) == 1
    assert bm25.normalize_sparse_query(normalized) == normalized


@pytest.mark.parametrize("query", ["Self-Projected Authority", "自我投射權威"])
def test_self_projected_authority_does_not_gain_a_self_center_alias(query: str) -> None:
    normalized = bm25.normalize_sparse_query(query)
    assert "Self-Projected Authority" in normalized
    assert not re.search(r"(?<!\w)G(?!\w)", normalized)
    assert bm25.normalize_sparse_query(normalized) == normalized


def test_normalization_preserves_unknown_words_and_avoids_synthetic_tokens() -> None:
    query = "Unusual curiosity Gate 42 Channel 42-53 Profile 4/6"
    normalized = bm25.normalize_sparse_query(query)
    assert "Unusual curiosity" in normalized
    assert not any(term in normalized for term in ("gate_", "channel_", "profile_"))
    assert normalized.count("Hexagram 42") == 1
    assert bm25.normalize_sparse_query(normalized) == normalized
    assert bm25.normalize_sparse_query("   ") == ""
    assert "Sacral" not in bm25.normalize_sparse_query("Splenic Authority")


def test_real_bm25_persistence_reload_and_safe_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = _nodes(tmp_path)
    directory = tmp_path / "bm25"
    bm25.build_and_persist_bm25(nodes, directory, "test-ingestion")
    sidecar = json.loads((directory / "identity.json").read_text())
    assert sidecar["ingestion_id"] == "test-ingestion"
    assert sidecar["chunk_count"] == len(nodes)
    def forbidden(*args, **kwargs):
        pytest.fail("Reload/retrieval must not read PDFs or construct embeddings")
    monkeypatch.setattr("human_design.rag.ingestion.load_pdfs", forbidden)
    monkeypatch.setattr("human_design.rag.embeddings.create_openai_embedding_model", forbidden)
    monkeypatch.setattr(hybrid, "load_manifest", forbidden)
    monkeypatch.setattr(hybrid, "load_nodes", forbidden)
    (tmp_path / "manifest.json").write_text("not a manifest")
    loaded = bm25.load_bm25_retriever(directory, "test-ingestion")
    assert loaded.ingestion_id == "test-ingestion"
    assert loaded.chunk_ids == tuple(node.node_id for node in nodes)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        loaded.ingestion_id = "changed"
    results = loaded.retrieve("42號閘門", 20)
    assert len(results) == 3
    assert results[0].chunk_id == nodes[0].node_id
    assert results[0].text == nodes[0].text
    assert [result.sparse_rank for result in results] == [1, 2, 3]
    raw = loaded._retriever.retrieve(bm25.normalize_sparse_query("42號閘門"))
    assert results[0].sparse_score == raw[0].score
    for result in results:
        assert result.dense_rank is result.dense_score is result.rrf_score is result.rerank_score is None
        assert result.source_file == "synthetic.pdf"
        assert result.source_relpath == "synthetic.pdf"
        assert result.document_title == "Synthetic reference"
        assert "private_cache_path" not in result.metadata
        assert "source_path" not in result.metadata
        assert str(tmp_path) not in json.dumps(dict(result.metadata))
    assert loaded.retrieve("Gate 1", 1)[0].chunk_id == nodes[1].node_id
    assert loaded.retrieve("", 1) == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_sparse_top_k_must_be_positive_integer(tmp_path: Path, top_k: object) -> None:
    directory = tmp_path / "bm25"
    bm25.build_and_persist_bm25(_nodes(tmp_path), directory, "identity")
    loaded = bm25.load_bm25_retriever(directory, "identity")
    with pytest.raises(ValueError, match="top_k"):
        loaded.retrieve("Gate 42", top_k)


@pytest.mark.parametrize("field", ["ingestion_id", "chunk_count", "chunk_ids"])
def test_sidecar_tampering_is_rejected(tmp_path: Path, field: str) -> None:
    directory = tmp_path / "bm25"
    bm25.build_and_persist_bm25(_nodes(tmp_path), directory, "identity")
    path = directory / "identity.json"
    payload = json.loads(path.read_text())
    payload[field] = {"ingestion_id": "wrong", "chunk_count": 100, "chunk_ids": []}[field]
    path.write_text(json.dumps(payload))
    with pytest.raises(hybrid.HybridIndexError, match="BM25") as exc:
        bm25.load_bm25_retriever(directory, "identity")
    assert str(tmp_path) not in str(exc.value)


def test_wrong_expected_ingestion_fails_before_library_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "bm25"
    bm25.build_and_persist_bm25(_nodes(tmp_path), directory, "identity")
    monkeypatch.setattr(bm25.BM25Retriever, "from_persist_dir", lambda *a, **k: pytest.fail("must fail first"))
    with pytest.raises(hybrid.HybridIndexError, match="ingestion"):
        bm25.load_bm25_retriever(directory, "wrong")


def test_missing_or_nonempty_bm25_is_safe(tmp_path: Path) -> None:
    with pytest.raises(hybrid.HybridIndexError, match="BM25"):
        bm25.load_bm25_retriever(tmp_path / "missing", "identity")
    assert not (tmp_path / "missing").exists()
    nodes = _nodes(tmp_path)
    directory = tmp_path / "bm25"
    directory.mkdir()
    (directory / "keep").write_text("untouched")
    with pytest.raises(hybrid.HybridIndexError, match="non-empty"):
        bm25.build_and_persist_bm25(nodes, directory, "identity")
    assert (directory / "keep").read_text() == "untouched"
