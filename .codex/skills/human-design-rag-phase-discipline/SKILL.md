---
name: human-design-rag-phase-discipline
description: |
  Use when planning or implementing any Human Design project feature.
  The active phase is Phase 2: local BodyGraph Vision extraction.
  Use this skill to prevent scope creep into reading generation, RAG answers,
  web apps, cloud deployment, OCR, computer-vision training, or unrelated infrastructure.
  Preserve Phase 1 as a stable local PDF-to-Chroma retrieval baseline.
---

# Human Design RAG Phase Discipline

## Purpose

Use this skill to keep work within the active project phase.

The current project phase is Phase 2: Local BodyGraph Vision Extraction.

Phase 1 is complete and protected as a stable baseline.

## Phase 1 Protected Baseline

Phase 1 pipeline:

    Local PDFs
       ↓
    PyMuPDFReader
       ↓
    Embedded text-only extraction
       ↓
    SentenceSplitter(chunk_size=800, chunk_overlap=80)
       ↓
    OpenAIEmbedding(text-embedding-3-small)
       ↓
    Persistent Chroma vector store
       ↓
    Retrieval smoke test

Do not modify Phase 1 PDF ingestion, chunking, embeddings, Chroma persistence, or retrieval architecture unless an active Phase 2 task explicitly requires a small compatible integration.

Phase 2 must not depend on Phase 1 Chroma storage for default tests.

Phase 1 retrieval is not part of Phase 2 runtime behavior.

## Phase 2 Scope

Phase 2 is local BodyGraph Vision extraction only.

In scope:

- Local Human Design BodyGraph or chart images
- Vision API extraction of raw visible facts only
- Strict JSON parsing
- Typed raw extraction models
- Canonical normalization of centers, channels, gates, and planetary activations
- Deterministic Python derivation of:
  - active gates
  - active channels
  - derived defined centers
  - profile
  - type
  - authority
  - definition
  - strategy
  - not-self theme
  - signature
- Parser issue preservation
- Validation with machine-readable warning codes
- Golden-label evaluation
- Mocked and offline tests
- Local CLI smoke testing
- Opt-in real Vision API testing only

## Fixed Phase 2 Architecture

    Local BodyGraph image
       ↓
    Vision raw fact extraction
       ↓
    Strict JSON parser
       ↓
    Raw facts plus confidence and uncertainty
       ↓
    Deterministic Python interpreter
       ↓
    Derived chart data
       ↓
    Validation warnings
       ↓
    Evaluation and local CLI smoke test

## Phase 2 Architecture Rules

1. Vision models extract raw visible facts only.

2. Vision models must not directly determine:
   - type
   - authority
   - profile
   - definition
   - strategy
   - not-self theme
   - signature

3. Deterministic Python rules derive final chart concepts.

4. Personality and Design planetary activation columns are the primary source of truth for active gates.

5. Active channels are derived only from canonical channels whose two endpoint gates are active.

6. Derived defined centers come from deterministic active-channel endpoints.

7. Visual centers, visually active gates, and visible colored channels are supporting evidence for validation only.

8. Invalid JSON and missing required structural fields are parser errors.

9. Recoverable Vision uncertainty, malformed observed values, normalization events, and visual-versus-derived disagreements become structured validation issues.

10. Phase 2 must not use an LLM to fill in missing deterministic facts or silently guess unsupported authority cases.

11. Unsupported or ambiguous deterministic cases must return a conservative result such as "Unknown" or "Needs Review" with a structured warning.

## Explicitly Out of Scope for Phase 2

Do not add:

- Human Design reading generation
- RAG answer generation
- Calling Phase 1 retrieval to generate readings
- Claude or GPT reading generation
- Streamlit
- FastAPI
- Web app
- Production API
- User accounts
- AWS
- S3
- Lambda
- API Gateway
- EC2
- SageMaker
- Cloud deployment
- Databases
- SQL database
- Pinecone
- OpenSearch
- LangChain
- OCR for arbitrary PDFs
- LlamaParse
- Parsing BodyGraph diagrams embedded inside Phase 1 PDFs
- Training computer vision models
- YOLO or object-detection training
- Manual annotation UI
- Astrology calculation from birth date, time, or location
- Multi-provider Vision abstractions unless explicitly requested
- New dependencies unless required by the active Phase 2 task

## Decision Protocol

When asked to add a feature:

1. Determine whether the requested work belongs to Phase 2 local BodyGraph Vision extraction.

2. If it is in scope, implement or plan it according to the Phase 2 architecture.

3. If it modifies Phase 1, allow it only when the active Phase 2 task explicitly requires a small compatible integration.

4. If it is reading generation, RAG answer generation, web or API work, cloud deployment, OCR, training, or another future phase, do not implement it.

5. Record future-phase ideas only as a brief note when useful. Do not create implementation tasks for them.

6. If the request is ambiguous between raw extraction and final Human Design interpretation, preserve the raw-extraction versus deterministic-interpreter boundary.

7. If deterministic rules cannot safely resolve a case, use a conservative warning-based result rather than an LLM guess.

## Common Mistakes to Reject

Reject these unless the user explicitly changes Phase 2 scope:

- "Let the Vision model directly infer Type and Authority."
- "Ask GPT to correct missing BodyGraph activations."
- "Use visible colored channels as final derived channels."
- "Use visual centers directly to determine Type."
- "Add RAG reading generation now."
- "Connect Phase 1 retrieval to the Phase 2 CLI."
- "Let's also add Streamlit."
- "Let's upload chart images to S3."
- "Let's add a FastAPI endpoint."
- "Let's use Pinecone or OpenSearch."
- "Let's add OCR for all PDFs."
- "Let's train YOLO to detect centers."
- "Let's add a manual chart annotation interface."
- "Let's add a second Vision provider before the first local pipeline is validated."

These may be future-phase ideas, but they must not be implemented during Phase 2 unless the user explicitly changes scope.

## Phase 2 Completion Standard

Phase 2 is complete only when:

- Raw Vision extraction is limited to raw visible facts.
- Parser behavior is tested offline.
- Deterministic interpreter derives active gates, channels, centers, profile, type, authority, definition, strategy, not-self theme, and signature conservatively.
- Validation produces machine-readable warnings.
- Evaluation compares predictions with golden labels.
- Mocked CLI extraction works without API keys.
- Default tests do not call paid APIs.
- Real Vision calls require explicit opt-in.
- Phase 1 tests continue to pass.
- No private images, raw base64 content, API keys, or private Vision responses are committed.