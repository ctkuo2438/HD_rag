# PDF ingestion and hybrid indexing

This replaces the completed Phase 1 task plan with the current ingestion contract. Setup and paid-provider opt-ins are in the [README](../README.md).

## Current flow

```text
Local PDFs -> PyMuPDFReader -> SentenceSplitter
  -> canonical nodes -> Chroma + BM25 -> verified manifest
```

- `rag/ingestion.py` discovers PDFs in deterministic order and extracts embedded text with source/page metadata. Scanned images are not OCR input.
- `rag/chunking.py` uses SentenceSplitter with defaults of 800 tokens and 80 overlap. It preserves page/source metadata, embedding model and ingestion version.
- `rag/embeddings.py` creates the configured OpenAI embedding client. Real build/query callers enforce the opt-in and key before construction.
- `rag/hybrid_index.py` coordinates the current build. `rag/vector_store.py` only opens a Chroma client at an explicit path.

The standalone Phase 1 ingestion/query scripts and their old Chroma configuration have been retired. PDF loading, chunking, extraction reports and embedding factories remain shared by the current system.

## Storage and identity

`HD_RAG_INDEX_DIR` defaults to `storage/hybrid_v1`:

```text
manifest.json
nodes.jsonl
chroma/
bm25/
```

Chroma and BM25 receive the same canonical node objects from one build. A chunk ID is SHA-256 of compact deterministic JSON containing the PDF-byte SHA-256, stable page key, zero-based per-page chunk ordinal and normalized text. Text normalizes CRLF/CR to LF, applies Unicode NFC and trims only outer whitespace. Blank chunks are excluded before assigning ordinals.

The LlamaIndex node ID matches the chunk ID. The deterministic ingestion identity excludes itself from its hash and is shared by the manifest and both indexes. Reload validates canonical IDs, metadata and counts. Existing v1 index format and identity algorithms remain unchanged.

## Build and reload rules

- `scripts/build_hybrid_index.py` requires explicit real embeddings opt-in and an OpenAI key before loading PDFs or constructing providers.
- A build requires an empty target directory. Existing files are never overwritten or deleted.
- When the PDFs, chunk settings or embedding model change, build into a fresh `HD_RAG_INDEX_DIR` and use that root for subsequent questions.
- Querying reloads existing indexes without reading PDFs or rebuilding vectors. Both Chroma and BM25 must be available and share an ingestion identity.
- BM25 preserves canonical IDs and raw sparse scores. Dense retrieval preserves raw dense scores. Their ranks are 1-based.
- Private PDF text, generated indexes and internal filesystem provenance stay out of Git and public citations.

## Regression coverage

`test_ingestion.py`, `test_chunking.py` and `test_embeddings.py` cover the shared ingestion components. `test_hybrid_index.py`, `test_bm25.py` and `test_retriever.py` cover persistence, identity checks, provider gates and reload behavior. Run the [offline verification](../README.md#default-verification).
