"""Offline canonical storage and coordinated Phase 3 index contracts."""

import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from human_design.rag import hybrid_index as hybrid
from human_design.rag.config import load_config
from human_design.rag.hybrid_index import build_hybrid_index, verify_hybrid_index


def _corpus(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "b.pdf").write_bytes(b"synthetic PDF B")
    (pdf_dir / "a.pdf").write_bytes(b"synthetic PDF A")
    config = replace(load_config(env={}), pdf_dir=pdf_dir, index_dir=tmp_path / "index")
    sources = hybrid.fingerprint_sources(pdf_dir)
    chunks = [TextNode(text=text, metadata={
        "source_path": str(pdf_dir / filename), "source_file": filename,
        "page_label": page, "page_number": 1, "document_title": "Synthetic reference",
        "chunk_size": 800, "chunk_overlap": 80,
        "embedding_model": config.embedding_model, "ingestion_version": "v1",
    }) for filename, page, text in (
        ("a.pdf", "1", " Gate 42 completes cycles.\r\nCafe\u0301. "),
        ("a.pdf", "1", "Sacral Authority responds."),
        ("b.pdf", "2", "Profile 4/6 has two lines."),
    )]
    nodes = hybrid.create_canonical_nodes(chunks, sources, pdf_dir)
    manifest = hybrid.create_manifest(nodes, sources, config)
    return config, nodes, manifest


def test_exact_chunk_id_algorithm_and_text_normalization() -> None:
    raw = " \r\nCafe\u0301\rline\n  internal  spaces\tremain \n"
    expected = "Café\nline\n  internal  spaces\tremain"
    assert hybrid.normalize_chunk_text(raw) == expected
    payload = '["sha","label:一",0,"Café\\nline\\n  internal  spaces\\tremain"]'
    assert hybrid.compute_chunk_id("sha", "label:一", 0, raw) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(("args", "changed"), [
    (("sha", "none", 0, "text"), ("sha2", "none", 0, "text")),
    (("sha", "none", 0, "text"), ("sha", "label:1", 0, "text")),
    (("sha", "none", 0, "text"), ("sha", "none", 1, "text")),
    (("sha", "none", 0, "text"), ("sha", "none", 0, "text!")),
    (("a|b", "c", 0, "x"), ("a", "b|c", 0, "x")),
])
def test_every_chunk_identity_input_matters(args: tuple, changed: tuple) -> None:
    assert hybrid.compute_chunk_id(*args) != hybrid.compute_chunk_id(*changed)


@pytest.mark.parametrize(("metadata", "expected"), [
    ({"page_label": "iv", "page_number": 4}, "label:iv"),
    ({"page_number": 4}, "number:4"),
    ({"page_label": "", "page_number": 4}, "number:4"),
    ({}, "none"),
])
def test_stable_page_key_fallback(metadata: dict, expected: str) -> None:
    assert hybrid.stable_page_key(metadata) == expected


def test_source_fingerprints_hash_bytes_in_discovery_order(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    assert [s.source_relpath for s in manifest.source_fingerprints] == ["a.pdf", "b.pdf"]
    assert manifest.source_fingerprints[0].source_sha256 == hashlib.sha256(
        b"synthetic PDF A"
    ).hexdigest()
    assert [node.metadata["chunk_index"] for node in nodes] == [0, 1, 0]
    assert all(node.node_id == node.metadata["chunk_id"] for node in nodes)
    assert nodes[0].text == "Gate 42 completes cycles.\nCafé."
    before = nodes[0].node_id
    (config.pdf_dir / "a.pdf").write_bytes(b"changed bytes")
    sources = hybrid.fingerprint_sources(config.pdf_dir)
    assert hybrid.create_canonical_nodes(nodes, sources, config.pdf_dir)[0].node_id != before


@pytest.mark.parametrize("blank", ["", " \t\r\n", "\u00a0\u3000"])
def test_canonical_nodes_skip_blank_text_without_changing_content_ids(tmp_path: Path, blank: str) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    empty = TextNode(text=blank, metadata=dict(nodes[0].metadata))
    chunks = [empty, nodes[0], empty, *nodes[1:], empty]
    actual = hybrid.create_canonical_nodes(chunks, manifest.source_fingerprints, config.pdf_dir)
    assert [(node.node_id, node.text) for node in actual] == [(node.node_id, node.text) for node in nodes]
    assert [node.metadata["chunk_index"] for node in actual] == [0, 1, 0]
    assert hybrid.create_canonical_nodes([empty], manifest.source_fingerprints, config.pdf_dir) == []


def test_duplicate_source_bytes_fail_with_only_relative_names(tmp_path: Path) -> None:
    (tmp_path / "first.pdf").write_bytes(b"same")
    (tmp_path / "second.pdf").write_bytes(b"same")
    with pytest.raises(hybrid.HybridIndexError, match="duplicate source") as exc:
        hybrid.fingerprint_sources(tmp_path)
    assert "first.pdf" in str(exc.value) and "second.pdf" in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


def test_duplicate_fingerprints_rejected_before_creating_nodes(tmp_path: Path) -> None:
    sources = (
        hybrid.SourceFingerprint("first.pdf", "a" * 64),
        hybrid.SourceFingerprint("second.pdf", "a" * 64),
    )
    with pytest.raises(hybrid.HybridIndexError, match="duplicate source"):
        hybrid.create_canonical_nodes([object()], sources, tmp_path)


def test_canonical_json_is_compact_sorted_and_keeps_ordered_lists() -> None:
    assert hybrid.canonical_json({"z": [2, 1], "a": "一"}) == '{"a":"一","z":[2,1]}'
    assert hybrid.canonical_json({"a": "一", "z": [2, 1]}) == hybrid.canonical_json(
        {"z": [2, 1], "a": "一"}
    )


def test_persistence_is_byte_identical_and_round_trips_without_source_files(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    first, second = config.index_dir, tmp_path / "second"
    hybrid.persist_canonical_index(first, nodes, manifest)
    hybrid.persist_canonical_index(second, nodes, manifest)
    for name in ("nodes.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for source in config.pdf_dir.iterdir():
        source.unlink()
    loaded_manifest = hybrid.load_manifest(first / "manifest.json")
    loaded = hybrid.load_nodes(first / "nodes.jsonl", loaded_manifest)
    assert loaded_manifest == manifest
    assert [(n.node_id, n.text, n.metadata) for n in loaded] == [
        (n.node_id, n.text, n.metadata) for n in nodes
    ]
    assert all(n.get_content(metadata_mode="embed") == n.text for n in loaded)


def test_manifest_identity_hash_excludes_itself(tmp_path: Path) -> None:
    _, nodes, manifest = _corpus(tmp_path)
    payload = asdict(manifest)
    ingestion_id = payload.pop("ingestion_id")
    assert ingestion_id == hashlib.sha256(hybrid.canonical_json(payload).encode()).hexdigest()
    assert set(payload) == {"schema_version", "ingestion_version", "corpus_fingerprint",
                            "source_fingerprints", "chunk_count", "chunk_size", "chunk_overlap",
                            "embedding_model", "collection_name", "chunk_id_algorithm"}
    assert manifest.chunk_count == len(nodes)
    assert manifest.collection_name == "human_design_hybrid_v1"
    assert manifest.chunk_id_algorithm == hybrid.CHUNK_ID_ALGORITHM


@pytest.mark.parametrize("field", [
    "schema_version", "ingestion_version", "corpus_fingerprint", "source_fingerprints",
    "chunk_count", "chunk_size", "chunk_overlap", "embedding_model", "collection_name",
    "chunk_id_algorithm", "ingestion_id",
])
def test_manifest_tampering_fails_reload(tmp_path: Path, field: str) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    hybrid.persist_canonical_index(config.index_dir, nodes, manifest)
    path = config.index_dir / "manifest.json"
    payload = json.loads(path.read_text())
    original = payload[field]
    payload[field] = original + 1 if type(original) is int else "tampered"
    if field != "ingestion_id":
        altered = {k: v for k, v in payload.items() if k != "ingestion_id"}
        assert hashlib.sha256(hybrid.canonical_json(altered).encode()).hexdigest() != manifest.ingestion_id
    path.write_text(json.dumps(payload))
    with pytest.raises(hybrid.HybridIndexError, match="manifest") as exc:
        hybrid.load_manifest(path)
    assert str(tmp_path) not in str(exc.value)


def test_duplicate_ids_fail_before_persistence(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    with pytest.raises(hybrid.HybridIndexError, match=nodes[0].node_id):
        hybrid.persist_canonical_index(config.index_dir, [nodes[0], nodes[0]], manifest)
    assert not config.index_dir.exists()


@pytest.mark.parametrize("content", ["not json\n", "{}\n", "[]\n", '{"chunk_id":NaN}\n'])
def test_malformed_jsonl_is_safe(tmp_path: Path, content: str) -> None:
    path = tmp_path / "nodes.jsonl"
    path.write_text(content)
    with pytest.raises(hybrid.HybridIndexError, match="nodes.jsonl") as exc:
        hybrid.load_nodes(path)
    assert str(tmp_path) not in str(exc.value)


def test_jsonl_tampered_text_duplicate_id_and_count_mismatch(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    hybrid.persist_canonical_index(config.index_dir, nodes, manifest)
    path = config.index_dir / "nodes.jsonl"
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n")
    with pytest.raises(hybrid.HybridIndexError, match="count"):
        hybrid.load_nodes(path, manifest)
    path.write_text(lines[0] + "\n" + lines[0] + "\n")
    with pytest.raises(hybrid.HybridIndexError, match="duplicate canonical"):
        hybrid.load_nodes(path)
    row = json.loads(lines[0])
    row["text"] = "Tampered text"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(hybrid.HybridIndexError, match="identity"):
        hybrid.load_nodes(path)


def test_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    config.index_dir.mkdir()
    marker = config.index_dir / "keep"
    marker.write_text("untouched")
    with pytest.raises(hybrid.HybridIndexError, match="non-empty"):
        hybrid.persist_canonical_index(config.index_dir, nodes, manifest)
    assert marker.read_text() == "untouched"
    assert not (config.index_dir / "nodes.jsonl").exists()


def test_safe_source_metadata_projection_drops_private_paths(tmp_path: Path) -> None:
    _, nodes, _ = _corpus(tmp_path)
    metadata = dict(nodes[0].metadata, private_cache_path="private/cache", arbitrary={"key": "secret"})
    projected = hybrid.safe_source_metadata(metadata)
    assert projected["source_file"] == "a.pdf"
    assert projected["source_relpath"] == "a.pdf"
    assert projected["document_title"] == "Synthetic reference"
    assert "source_path" not in projected and "private_cache_path" not in projected
    assert "arbitrary" not in projected and str(tmp_path) not in json.dumps(projected)
    assert "document_title" not in hybrid.safe_source_metadata({"document_title": "/private/book"})


def test_manifest_rejects_nodes_from_different_source_fingerprints(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    sources = (replace(manifest.source_fingerprints[0], source_sha256="f" * 64),
               manifest.source_fingerprints[1])
    with pytest.raises(hybrid.HybridIndexError, match="source"):
        hybrid.create_manifest(nodes, sources, config)


def test_canonical_reload_rejects_unsafe_relative_source(tmp_path: Path) -> None:
    config, nodes, manifest = _corpus(tmp_path)
    hybrid.persist_canonical_index(config.index_dir, nodes, manifest)
    path = config.index_dir / "nodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["metadata"]["source_relpath"] = "/private/reference.pdf"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(hybrid.HybridIndexError, match="identity"):
        hybrid.load_nodes(path, manifest)


@pytest.fixture
def build_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "synthetic.pdf").write_bytes(b"fake PDF bytes for fingerprint only")
    documents = [Document(text=text, metadata={
        "source_path": str(pdf_dir / "synthetic.pdf"), "file_name": "synthetic.pdf",
        "page_label": str(i), "page_number": i,
    }) for i, text in enumerate(("Gate 42 completes cycles.", "Gate 1 expresses creativity."), 1)]
    config = replace(load_config(env={}), pdf_dir=pdf_dir, index_dir=tmp_path / "hybrid",
                     real_embeddings=True, openai_api_key="fake-test-key")
    monkeypatch.setattr(hybrid, "load_pdfs", lambda path: documents)
    monkeypatch.setattr(hybrid, "create_openai_embedding_model_from_config", lambda config: MockEmbedding(embed_dim=8))
    return config


def test_existing_dotenv_legacy_settings_cannot_redirect_or_block_hybrid_index(build_setup, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"HD_RAG_PDF_DIR={build_setup.pdf_dir}\n"
        f"HD_RAG_INDEX_DIR={build_setup.index_dir}\n"
        f"HD_RAG_CHROMA_DIR={build_setup.index_dir}\n"
        "HD_RAG_COLLECTION=unused-old-collection\n"
        "HD_RAG_REAL_EMBEDDINGS=1\nOPENAI_API_KEY=fake-test-key\n"
    )
    config = load_config(env={}, env_file=env_file)
    manifest = build_hybrid_index(config)
    assert manifest.collection_name == "human_design_hybrid_v1"
    assert verify_hybrid_index(config) == manifest
    assert (config.index_dir / "chroma").is_dir()
    assert (config.index_dir / "bm25").is_dir()


@pytest.mark.parametrize("include_blank_pages", [False, True])
def test_coordinated_build_shares_objects_ids_and_identity(
    build_setup, monkeypatch: pytest.MonkeyPatch, include_blank_pages: bool,
) -> None:
    from human_design.rag import bm25
    from human_design.rag.vector_store import create_chroma_client
    config = build_setup
    if include_blank_pages:
        documents = hybrid.load_pdfs(config.pdf_dir)
        documents.extend(Document(text=text, metadata={
            **documents[0].metadata, "page_label": str(page), "page_number": page,
        }) for page, text in enumerate(("", " \t\r\n"), 3))
    seen = {}
    real_dense, real_sparse = hybrid.build_phase3_chroma, bm25.build_and_persist_bm25
    def dense(nodes, *args, **kwargs):
        seen["dense"] = nodes
        return real_dense(nodes, *args, **kwargs)
    def sparse(nodes, directory, ingestion_id):
        seen["sparse"] = nodes
        seen["ingestion_id"] = ingestion_id
        return real_sparse(nodes, directory, ingestion_id)
    monkeypatch.setattr(hybrid, "build_phase3_chroma", dense)
    monkeypatch.setattr(bm25, "build_and_persist_bm25", sparse)
    manifest = build_hybrid_index(config)
    assert seen["dense"] is seen["sparse"]
    assert all(a is b for a, b in zip(seen["dense"], seen["sparse"], strict=True))
    assert seen["ingestion_id"] == manifest.ingestion_id
    canonical = hybrid.load_nodes(config.index_dir / "nodes.jsonl", manifest)
    ids = [node.node_id for node in canonical]
    assert len(ids) == 2
    collection = create_chroma_client(config.index_dir / "chroma").get_collection(hybrid.COLLECTION_NAME)
    assert set(collection.get(include=[])["ids"]) == set(ids)
    sparse_adapter = bm25.load_bm25_retriever(config.index_dir / "bm25", manifest.ingestion_id)
    assert sparse_adapter.chunk_ids == tuple(ids)
    assert collection.count() == manifest.chunk_count == len(ids)
    for name in ("embedding_model", "ingestion_id", "schema_version", "ingestion_version", "corpus_fingerprint", "chunk_count"):
        assert collection.metadata[name] == getattr(manifest, name)
    assert verify_hybrid_index(config) == manifest


def test_all_blank_pages_fail_before_embedding_or_storage(build_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_setup
    documents = hybrid.load_pdfs(config.pdf_dir)
    monkeypatch.setattr(hybrid, "load_pdfs", lambda path: [
        Document(text=" \t\r\n", metadata=dict(documents[0].metadata)),
    ])

    def forbidden(*args, **kwargs):
        pytest.fail("Blank documents must not construct embeddings or indexes")

    monkeypatch.setattr(hybrid, "create_openai_embedding_model_from_config", forbidden)
    with pytest.raises(hybrid.HybridIndexError, match="No canonical text chunks"):
        build_hybrid_index(config)
    assert not config.index_dir.exists()


@pytest.mark.parametrize("builder", ["dense", "sparse"])
@pytest.mark.parametrize("alteration", ["reverse", "drop", "identity"])
def test_build_rejects_different_builder_nodes_or_identity(
    build_setup, monkeypatch: pytest.MonkeyPatch, builder: str, alteration: str,
) -> None:
    from human_design.rag import bm25
    if builder == "dense":
        real = hybrid.build_phase3_chroma
        def changed(nodes, config, manifest, embed_model):
            if alteration == "identity":
                manifest = replace(manifest, ingestion_id="wrong")
            else:
                nodes = list(reversed(nodes)) if alteration == "reverse" else nodes[:1]
            return real(nodes, config, manifest, embed_model)
        monkeypatch.setattr(hybrid, "build_phase3_chroma", changed)
    else:
        real = bm25.build_and_persist_bm25
        def changed(nodes, directory, ingestion_id):
            if alteration == "identity":
                ingestion_id = "wrong"
            else:
                nodes = list(reversed(nodes)) if alteration == "reverse" else nodes[:1]
            return real(nodes, directory, ingestion_id)
        monkeypatch.setattr(bm25, "build_and_persist_bm25", changed)
    with pytest.raises(hybrid.HybridIndexError, match="identity|count|manifest"):
        build_hybrid_index(build_setup)


@pytest.mark.parametrize("mode", ["disabled", "no-key", "nonempty"])
def test_build_gates_precede_loading_and_provider_construction(
    build_setup, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    config = build_setup
    if mode == "disabled":
        config = replace(config, real_embeddings=False)
    elif mode == "no-key":
        config = replace(config, openai_api_key=None)
    else:
        config.index_dir.mkdir()
        (config.index_dir / "keep").write_text("untouched")
    def forbidden(*args, **kwargs):
        pytest.fail("Gate must run before reading PDFs or constructing providers/storage")
    for name in ("load_pdfs", "create_openai_embedding_model_from_config", "build_phase3_chroma"):
        monkeypatch.setattr(hybrid, name, forbidden)
    with pytest.raises(hybrid.HybridIndexError):
        build_hybrid_index(config)
    if mode == "nonempty":
        assert (config.index_dir / "keep").read_text() == "untouched"
    else:
        assert not config.index_dir.exists()


@pytest.mark.parametrize("artifact", ["nodes", "chroma", "bm25"])
def test_post_build_verification_detects_missing_index_content(build_setup, artifact: str) -> None:
    config = build_setup
    build_hybrid_index(config)
    if artifact == "nodes":
        path = config.index_dir / "nodes.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    elif artifact == "chroma":
        from human_design.rag.vector_store import create_chroma_client
        collection = create_chroma_client(config.index_dir / "chroma").get_collection(hybrid.COLLECTION_NAME)
        collection.delete(ids=collection.get(include=[])["ids"][:1])
    else:
        path = config.index_dir / "bm25" / "identity.json"
        payload = json.loads(path.read_text())
        payload["chunk_count"] += 1
        path.write_text(json.dumps(payload))
    with pytest.raises(hybrid.HybridIndexError, match="count|identity"):
        verify_hybrid_index(config)


def _build_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_hybrid_index.py"
    spec = importlib.util.spec_from_file_location("build_hybrid_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["disabled", "no-key"])
def test_build_script_gates_without_touching_storage(build_setup, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    script = _build_script()
    config = replace(build_setup, **({"real_embeddings": False} if mode == "disabled" else {"openai_api_key": None}))
    monkeypatch.setattr(script, "load_config", lambda: config)
    monkeypatch.setattr(script, "build_hybrid_index", lambda config: pytest.fail("build must not run"))
    with pytest.raises(SystemExit, match="HD_RAG_REAL_EMBEDDINGS|OPENAI_API_KEY"):
        script.main([])
    assert not config.index_dir.exists()


def test_build_script_output_is_safe_and_has_no_append_mode(build_setup, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    script = _build_script()
    monkeypatch.setattr(script, "load_config", lambda: build_setup)
    script.main([])
    output = capsys.readouterr().out
    assert "cost" in output and "chunks=2" in output
    assert build_setup.openai_api_key not in output
    assert str(build_setup.pdf_dir) not in output
    assert "Gate 42 completes cycles" not in output
    with pytest.raises(SystemExit):
        script.main(["--append"])


@pytest.mark.parametrize("field", [
    "embedding_model", "ingestion_id", "schema_version", "ingestion_version",
    "corpus_fingerprint", "chunk_count", "ordered_chunk_ids_sha256",
])
def test_dense_reload_rejects_chroma_identity_before_provider(
    build_setup, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    from human_design.rag import retriever
    from human_design.rag.vector_store import create_chroma_client
    config = build_setup
    build_hybrid_index(config)
    collection = create_chroma_client(config.index_dir / "chroma").get_collection(hybrid.COLLECTION_NAME)
    metadata = dict(collection.metadata)
    metadata[field] = "wrong" if isinstance(metadata[field], str) else metadata[field] + 1
    collection.modify(metadata=metadata)
    monkeypatch.setattr(retriever, "create_openai_embedding_model_from_config", lambda config: pytest.fail("no provider"))
    with pytest.raises(hybrid.HybridIndexError, match="identity"):
        retriever.load_dense_retriever(config)


def test_real_local_dense_reload_never_modifies_vectors(build_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    from human_design.rag.retriever import load_dense_retriever
    from human_design.rag.vector_store import create_chroma_client
    config = build_setup
    manifest = build_hybrid_index(config)
    collection = create_chroma_client(config.index_dir / "chroma").get_collection(hybrid.COLLECTION_NAME)
    before = collection.get(include=["documents", "metadatas"])
    def forbidden(*args, **kwargs):
        pytest.fail("Dense reload/retrieval must not append/delete/update vectors")
    for method in ("add", "upsert", "delete", "update", "modify"):
        monkeypatch.setattr(type(collection), method, forbidden)
    adapter = load_dense_retriever(config, embed_model=MockEmbedding(embed_dim=8))
    assert adapter.ingestion_id == manifest.ingestion_id
    results = adapter.retrieve("What does Gate 42 mean?", 2)
    assert {result.chunk_id for result in results} == set(before["ids"])
    assert [result.dense_rank for result in results] == [1, 2]
    assert collection.get(include=["documents", "metadatas"]) == before


def test_build_preserves_an_unrelated_collection(build_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    from human_design.rag.vector_store import create_chroma_client
    config = build_setup
    unrelated = create_chroma_client(config.index_dir.parent / "unrelated").create_collection("unrelated")
    unrelated.add(ids=["unrelated-id"], embeddings=[[0.1, 0.2]], documents=["Synthetic unrelated marker"])
    before = unrelated.get(include=["documents", "metadatas"])
    def restricted_client(path):
        assert path == config.index_dir / "chroma"
        return create_chroma_client(path)
    monkeypatch.setattr(hybrid, "create_chroma_client", restricted_client)
    build_hybrid_index(config)
    assert unrelated.get(include=["documents", "metadatas"]) == before


def test_same_count_but_different_chroma_ids_fails_cross_index_check(build_setup) -> None:
    from human_design.rag.vector_store import create_chroma_client
    config = build_setup
    manifest = build_hybrid_index(config)
    collection = create_chroma_client(config.index_dir / "chroma").get_collection(hybrid.COLLECTION_NAME)
    collection.delete(ids=collection.get(include=[])["ids"][:1])
    collection.add(ids=["different-id"], embeddings=[[0.0] * 8], documents=["Synthetic replacement"])
    assert collection.count() == manifest.chunk_count
    with pytest.raises(hybrid.HybridIndexError, match="identity"):
        verify_hybrid_index(config)
