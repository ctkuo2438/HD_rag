"""Deterministic canonical nodes and local Phase 3 corpus identity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from human_design.rag.chunking import chunk_documents
from human_design.rag.config import AppConfig
from human_design.rag.embeddings import create_openai_embedding_model_from_config
from human_design.rag.ingestion import build_text_extraction_report, discover_pdfs, load_pdfs
from human_design.rag.models import JsonValue, _copy_json_value
from human_design.rag.vector_store import create_chroma_client


SCHEMA_VERSION = 1
COLLECTION_NAME = "human_design_hybrid_v1"
CHUNK_ID_ALGORITHM = (
    "sha256(compact-json-utf8([source_sha256,page_key,chunk_index,normalized_text]));"
    "source_sha256=sha256(pdf-bytes);page_key=label:<nonempty-label>|number:<int>|none;"
    "chunk_index=zero-based-per-source-page;normalize=CRLF/CR-to-LF,NFC,strip;"
    "json=sort_keys:true,ensure_ascii:false,separators:(',',':'),allow_nan:false"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_IN_TEXT = re.compile(r"\bfile://|(?<![\w:/\\])(?:[A-Za-z]:[\\/]|[\\/])[^\s]+", re.I)


class HybridIndexError(ValueError):
    """Invalid or incompatible local hybrid artifacts; messages contain no local paths."""


@dataclass(frozen=True)
class SourceFingerprint:
    source_relpath: str
    source_sha256: str


@dataclass(frozen=True)
class HybridManifest:
    schema_version: int
    ingestion_id: str
    ingestion_version: str
    corpus_fingerprint: str
    source_fingerprints: tuple[SourceFingerprint, ...]
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    collection_name: str
    chunk_id_algorithm: str


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_chunk_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def stable_page_key(metadata: Mapping[str, object]) -> str:
    label = metadata.get("page_label")
    if isinstance(label, str) and label:
        return f"label:{label}"
    number = metadata.get("page_number")
    if type(number) is int:
        return f"number:{number}"
    return "none"


def compute_chunk_id(source_sha256: str, page_key: str, chunk_index: int, text: str) -> str:
    return _digest([source_sha256, page_key, chunk_index, normalize_chunk_text(text)])


def fingerprint_sources(pdf_dir: Path) -> tuple[SourceFingerprint, ...]:
    """Use shared PDF discovery; reject duplicate bytes before creating any nodes."""
    try:
        sources = tuple(SourceFingerprint(
            path.relative_to(pdf_dir).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest(),
        ) for path in discover_pdfs(pdf_dir))
    except OSError:
        raise HybridIndexError("Cannot discover or read configured PDF sources") from None
    _validate_sources(sources)
    return sources


def _validate_sources(sources: Sequence[SourceFingerprint]) -> None:
    hashes: dict[str, str] = {}
    names: set[str] = set()
    for source in sources:
        if not _safe_relative(source.source_relpath) or not _SHA256.fullmatch(source.source_sha256):
            raise HybridIndexError("Invalid source fingerprint or relative identity")
        if source.source_sha256 in hashes:
            raise HybridIndexError(
                f"duplicate source bytes: {hashes[source.source_sha256]}, {source.source_relpath}"
            )
        if source.source_relpath in names:
            raise HybridIndexError("Duplicate source relative identity")
        hashes[source.source_sha256] = source.source_relpath
        names.add(source.source_relpath)


def create_canonical_nodes(
    chunks: Sequence[BaseNode], sources: Sequence[SourceFingerprint], pdf_dir: Path,
) -> list[TextNode]:
    _validate_sources(sources)
    fingerprints = {source.source_relpath: source.source_sha256 for source in sources}
    ordinals: Counter[tuple[str, str]] = Counter()
    nodes: list[TextNode] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source_path = metadata.pop("source_path", None)
        if source_path is not None:
            try:
                relative = Path(source_path).resolve().relative_to(pdf_dir.resolve()).as_posix()
            except (ValueError, TypeError):
                raise HybridIndexError("Chunk source is outside the configured PDF directory") from None
        else:
            relative = metadata.get("source_relpath", metadata.get("source_file", metadata.get("file_name")))
        if relative not in fingerprints:
            raise HybridIndexError("Chunk source does not match discovered source fingerprints")
        text = normalize_chunk_text(chunk.get_content(metadata_mode=MetadataMode.NONE))
        # VectorStoreIndex skips empty content; both indexes must share that corpus.
        if not text:
            continue
        source_sha256 = fingerprints[relative]
        page_key = stable_page_key(metadata)
        ordinal = ordinals[source_sha256, page_key]
        ordinals[source_sha256, page_key] += 1
        chunk_id = compute_chunk_id(source_sha256, page_key, ordinal, text)
        metadata.update(chunk_id=chunk_id, source_sha256=source_sha256, source_relpath=relative,
                        source_file=PurePosixPath(relative).name, page_key=page_key, chunk_index=ordinal)
        nodes.append(_text_node(chunk_id, text, metadata))
    validate_canonical_nodes(nodes)
    return nodes


def _text_node(chunk_id: str, text: str, metadata: dict) -> TextNode:
    try:
        copied = _copy_json_value(metadata)
    except (TypeError, ValueError):
        raise HybridIndexError("Canonical metadata must be JSON-safe") from None
    # Index only the original passage text, never local diagnostics or synthetic lexical terms.
    return TextNode(id_=chunk_id, text=text, metadata=copied,
                    excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata))


def validate_canonical_nodes(nodes: Sequence[BaseNode], manifest: HybridManifest | None = None) -> None:
    seen: set[str] = set()
    for node in nodes:
        if not _SHA256.fullmatch(node.node_id):
            raise HybridIndexError("Invalid canonical chunk identity")
        if node.node_id in seen:
            raise HybridIndexError(f"duplicate canonical chunk ID: {node.node_id}")
        seen.add(node.node_id)
        metadata = node.metadata
        try:
            index = metadata["chunk_index"]
            if type(index) is not int or index < 0:
                raise ValueError
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            if (not _safe_relative(metadata["source_relpath"])
                    or not _SHA256.fullmatch(metadata["source_sha256"])
                    or metadata["source_file"] != PurePosixPath(metadata["source_relpath"]).name
                    or metadata["chunk_id"] != node.node_id or normalize_chunk_text(text) != text
                    or metadata["page_key"] != stable_page_key(metadata)
                    or compute_chunk_id(metadata["source_sha256"], metadata["page_key"], index, text) != node.node_id):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise HybridIndexError("Canonical node identity mismatch") from None
    if manifest is not None:
        if len(nodes) != manifest.chunk_count:
            raise HybridIndexError("Canonical node count does not match manifest")
        if _corpus_fingerprint(nodes, manifest.source_fingerprints) != manifest.corpus_fingerprint:
            raise HybridIndexError("Canonical corpus identity does not match manifest")
        sources = {source.source_relpath: source.source_sha256 for source in manifest.source_fingerprints}
        if any(sources.get(node.metadata["source_relpath"]) != node.metadata["source_sha256"] for node in nodes):
            raise HybridIndexError("Canonical source fingerprint does not match manifest")


def _corpus_fingerprint(nodes: Sequence[BaseNode], sources: Sequence[SourceFingerprint]) -> str:
    return _digest({"source_fingerprints": [asdict(source) for source in sources],
                    "chunk_ids": [node.node_id for node in nodes]})


def create_manifest(
    nodes: Sequence[BaseNode], sources: Sequence[SourceFingerprint], config: AppConfig,
) -> HybridManifest:
    validate_canonical_nodes(nodes)
    _validate_sources(sources)
    payload = dict(
        schema_version=SCHEMA_VERSION, ingestion_version=config.ingestion_version,
        corpus_fingerprint=_corpus_fingerprint(nodes, sources),
        source_fingerprints=[asdict(source) for source in sources], chunk_count=len(nodes),
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model, collection_name=COLLECTION_NAME,
        chunk_id_algorithm=CHUNK_ID_ALGORITHM,
    )
    manifest = HybridManifest(**(payload | {"source_fingerprints": tuple(sources), "ingestion_id": _digest(payload)}))
    validate_canonical_nodes(nodes, manifest)
    return manifest


def require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise HybridIndexError("Index target is non-empty; remove it intentionally or choose a fresh directory")


def persist_canonical_index(index_dir: Path, nodes: Sequence[BaseNode], manifest: HybridManifest) -> None:
    _manifest_from_dict(asdict(manifest))
    validate_canonical_nodes(nodes, manifest)
    require_empty_directory(index_dir)
    write_nodes(index_dir / "nodes.jsonl", nodes)
    write_manifest(index_dir / "manifest.json", manifest)


def _write_new(path: Path, content: str, artifact: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except OSError:
        raise HybridIndexError(f"Cannot create {artifact}; existing artifacts are never overwritten") from None


def write_nodes(path: Path, nodes: Sequence[BaseNode]) -> None:
    validate_canonical_nodes(nodes)
    content = "".join(canonical_json({"chunk_id": node.node_id,
        "text": node.get_content(metadata_mode=MetadataMode.NONE), "metadata": node.metadata}) + "\n" for node in nodes)
    _write_new(path, content, "nodes.jsonl")


def write_manifest(path: Path, manifest: HybridManifest) -> None:
    _manifest_from_dict(asdict(manifest))
    _write_new(path, canonical_json(asdict(manifest)) + "\n", "manifest.json")


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _read_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_strict_object, parse_constant=_reject_constant)


def load_nodes(path: Path, manifest: HybridManifest | None = None) -> list[TextNode]:
    try:
        nodes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _read_json(line)
            if not isinstance(row, dict) or set(row) != {"chunk_id", "text", "metadata"}:
                raise ValueError
            if not isinstance(row["chunk_id"], str) or not isinstance(row["text"], str) or not isinstance(row["metadata"], dict):
                raise ValueError
            nodes.append(_text_node(row["chunk_id"], row["text"], row["metadata"]))
    except (OSError, ValueError, TypeError):
        raise HybridIndexError("Cannot load malformed or missing nodes.jsonl") from None
    validate_canonical_nodes(nodes, manifest)
    return nodes


def _manifest_from_dict(payload: dict) -> HybridManifest:
    try:
        identity = {key: value for key, value in payload.items() if key != "ingestion_id"}
        if _digest(identity) != payload["ingestion_id"]:
            raise ValueError
        values = dict(payload)
        values["source_fingerprints"] = tuple(SourceFingerprint(**source) for source in payload["source_fingerprints"])
        manifest = HybridManifest(**values)
        _validate_sources(manifest.source_fingerprints)
        if (manifest.schema_version != SCHEMA_VERSION or manifest.collection_name != COLLECTION_NAME
                or manifest.chunk_id_algorithm != CHUNK_ID_ALGORITHM):
            raise ValueError
        for number in (manifest.chunk_count, manifest.chunk_size, manifest.chunk_overlap):
            if type(number) is not int or number < 0:
                raise ValueError
        if manifest.chunk_size <= manifest.chunk_overlap:
            raise ValueError
        if not _SHA256.fullmatch(manifest.corpus_fingerprint):
            raise ValueError
        return manifest
    except (KeyError, TypeError, ValueError):
        raise HybridIndexError("Invalid manifest structure or ingestion identity") from None


def load_manifest(path: Path) -> HybridManifest:
    try:
        payload = _read_json(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
    except (OSError, ValueError):
        raise HybridIndexError("Cannot read missing or malformed manifest.json") from None
    return _manifest_from_dict(payload)


def _safe_relative(value: str) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and not PurePosixPath(value).is_absolute()
            and not PureWindowsPath(value).anchor and ".." not in PureWindowsPath(value).parts
            and not value.lower().startswith("file:"))


def safe_source_metadata(metadata: Mapping[str, object]) -> dict[str, JsonValue]:
    """Allowlist useful provenance, excluding private/local diagnostic fields."""
    result: dict[str, JsonValue] = {}
    for name in ("chunk_id", "source_sha256", "source_file", "source_relpath", "document_title",
                 "page_label", "page_number", "chunk_size", "chunk_overlap", "embedding_model",
                 "ingestion_version"):
        value = metadata.get(name)
        if type(value) in (str, int, float, bool) or value is None:
            if isinstance(value, str):
                if name == "source_file":
                    value = PureWindowsPath(value).name
                if name == "source_relpath" and not _safe_relative(value):
                    continue
                if _PATH_IN_TEXT.search(value):
                    continue
            if value is not None:
                result[name] = value
    return result


def require_real_embeddings(config: AppConfig) -> None:
    """Call boundary shared by the hybrid build script and real dense loading."""
    if not config.real_embeddings:
        raise HybridIndexError("Real embeddings are disabled. Set HD_RAG_REAL_EMBEDDINGS=1 to opt in")
    if not config.openai_api_key or not config.openai_api_key.strip():
        raise HybridIndexError("OPENAI_API_KEY is required for real embeddings")


def _dense_metadata(nodes: Sequence[BaseNode], manifest: HybridManifest) -> dict[str, str | int]:
    return {"embedding_model": manifest.embedding_model, "ingestion_id": manifest.ingestion_id,
            "schema_version": manifest.schema_version, "ingestion_version": manifest.ingestion_version,
            "corpus_fingerprint": manifest.corpus_fingerprint, "chunk_count": manifest.chunk_count,
            "ordered_chunk_ids_sha256": _digest([node.node_id for node in nodes])}


def build_phase3_chroma(
    nodes: list[BaseNode], config: AppConfig, manifest: HybridManifest, embed_model: BaseEmbedding,
) -> None:
    """Build only a fresh Phase 3 collection from this run's canonical node objects."""
    _manifest_from_dict(asdict(manifest))
    validate_canonical_nodes(nodes, manifest)
    directory = config.index_dir / "chroma"
    require_empty_directory(directory)
    try:
        client = create_chroma_client(directory)
        collection = client.create_collection(
            name=COLLECTION_NAME, metadata=_dense_metadata(nodes, manifest), embedding_function=None,
        )
        store = ChromaVectorStore(chroma_collection=collection)
        VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store),
                         embed_model=embed_model, show_progress=False)
        _validate_dense_collection(collection, manifest, nodes)
    except HybridIndexError:
        raise
    except Exception:
        raise HybridIndexError("Cannot build Phase 3 Chroma index") from None


def _validate_dense_collection(collection: Any, manifest: HybridManifest, nodes: Sequence[BaseNode]) -> None:
    count = collection.count()
    if count == 0:
        raise HybridIndexError("Phase 3 Chroma collection is empty")
    if count != manifest.chunk_count:
        raise HybridIndexError("Phase 3 Chroma count does not match manifest")
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in _dense_metadata(nodes, manifest).items()):
        raise HybridIndexError("Phase 3 Chroma metadata/ingestion identity does not match manifest")
    ids = collection.get(include=[])["ids"]
    if len(ids) != len(nodes) or set(ids) != {node.node_id for node in nodes}:
        raise HybridIndexError("Phase 3 Chroma canonical chunk identity mismatch")


def open_phase3_chroma(config: AppConfig) -> tuple[ChromaVectorStore, HybridManifest]:
    """Strict existing-index open; never get-or-create a collection or repair metadata."""
    manifest = load_manifest(config.index_dir / "manifest.json")
    if manifest.embedding_model != config.embedding_model:
        raise HybridIndexError("Phase 3 manifest embedding model does not match configuration")
    nodes = load_nodes(config.index_dir / "nodes.jsonl", manifest)
    if any(node.metadata.get("ingestion_id") != manifest.ingestion_id for node in nodes):
        raise HybridIndexError("Canonical nodes ingestion identity does not match manifest")
    directory = config.index_dir / "chroma"
    if not directory.is_dir() or not any(directory.iterdir()):
        raise HybridIndexError("Phase 3 Chroma directory is missing or empty; build the index first")
    try:
        client = create_chroma_client(directory)
        collection = client.get_collection(name=manifest.collection_name, embedding_function=None)
        _validate_dense_collection(collection, manifest, nodes)
        return ChromaVectorStore(chroma_collection=collection), manifest
    except HybridIndexError:
        raise
    except Exception:
        raise HybridIndexError("Cannot open existing Phase 3 Chroma collection") from None


def verify_hybrid_index(config: AppConfig) -> HybridManifest:
    """Check persisted nodes, dense IDs/metadata, and sparse IDs/sidecar as one corpus."""
    from human_design.rag.bm25 import load_bm25_retriever

    _, manifest = open_phase3_chroma(config)
    nodes = load_nodes(config.index_dir / "nodes.jsonl", manifest)
    sparse = load_bm25_retriever(config.index_dir / "bm25", manifest.ingestion_id)
    if sparse.chunk_ids != tuple(node.node_id for node in nodes):
        raise HybridIndexError("BM25 ordered chunk identity does not match canonical manifest corpus")
    return manifest


def build_hybrid_index(config: AppConfig, *, embed_model: BaseEmbedding | None = None) -> HybridManifest:
    """One explicitly opted-in build using the shared PDF extraction and chunking."""
    from human_design.rag.bm25 import build_and_persist_bm25

    require_real_embeddings(config)
    require_empty_directory(config.index_dir)
    try:
        documents = load_pdfs(config.pdf_dir)
        report = build_text_extraction_report(config.pdf_dir, documents=documents)
        chunks, _ = chunk_documents(documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
                                    embedding_model=config.embedding_model, ingestion_version=config.ingestion_version)
    except Exception:
        raise HybridIndexError("Cannot extract or chunk configured PDF sources") from None
    sources = fingerprint_sources(config.pdf_dir)
    nodes = create_canonical_nodes(chunks, sources, config.pdf_dir)
    if not nodes or report.document_count == 0:
        raise HybridIndexError("No canonical text chunks available for hybrid indexing")
    manifest = create_manifest(nodes, sources, config)
    for node in nodes:
        node.metadata["ingestion_id"] = manifest.ingestion_id
        node.excluded_embed_metadata_keys.append("ingestion_id")
        node.excluded_llm_metadata_keys.append("ingestion_id")
    write_nodes(config.index_dir / "nodes.jsonl", nodes)
    try:
        embedding = embed_model if embed_model is not None else create_openai_embedding_model_from_config(config)
    except Exception:
        raise HybridIndexError("Cannot construct configured embedding provider") from None
    build_phase3_chroma(nodes, config, manifest, embedding)
    # Revalidate after each builder to catch accidental mutation before proceeding.
    validate_canonical_nodes(nodes, manifest)
    build_and_persist_bm25(nodes, config.index_dir / "bm25", manifest.ingestion_id)
    validate_canonical_nodes(nodes, manifest)
    write_manifest(config.index_dir / "manifest.json", manifest)
    return verify_hybrid_index(config)
