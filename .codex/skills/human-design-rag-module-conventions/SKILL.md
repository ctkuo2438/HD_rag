---
name: human-design-rag-module-conventions
description: |
  Use when creating or modifying Python modules under src/human_design/rag/
  or src/human_design/vision/. Core modules must stay interface-agnostic,
  local-first, typed, testable, and separated from CLI behavior and external
  providers. Use this when deciding file layout, models, imports, error handling,
  and module boundaries during Phase 2.
---

# Human Design RAG Module Conventions

## Purpose

Use this skill when creating or modifying Python modules for the Human Design project.

The active phase is Phase 2: local BodyGraph Vision extraction.

The goal is to keep:

- Phase 1 RAG modules stable and protected.
- Phase 2 Vision modules deterministic, testable, and local-first.
- CLI behavior separate from core logic.
- Vision provider code isolated from parsing and interpretation.
- Final chart concepts derived by deterministic Python rules.

## Project Module Boundaries

The project has two separate module areas:

    src/human_design/
    ├── rag/
    │   └── Phase 1 PDF ingestion and retrieval baseline
    └── vision/
        └── Phase 2 BodyGraph Vision extraction pipeline

Phase 1 and Phase 2 may share package-level utilities only when genuinely necessary.

Do not move, rename, or refactor Phase 1 RAG modules during Phase 2 unless an active task explicitly requires a small compatible integration.

Phase 2 must not import or depend on Phase 1 Chroma storage, retrieval, embeddings, or PDF ingestion for default behavior or tests.

## Expected Layout

    <repo-root>/
    ├── prompts/
    │   └── bodygraph_raw_extraction.txt
    ├── scripts/
    │   ├── ingest_pdfs.py
    │   ├── query_kb.py
    │   ├── extract_bodygraph.py
    │   └── evaluate_bodygraph_extraction.py
    └── src/
        └── human_design/
            ├── __init__.py
            ├── rag/
            │   ├── __init__.py
            │   ├── config.py
            │   ├── models.py
            │   ├── ingestion.py
            │   ├── chunking.py
            │   ├── embeddings.py
            │   ├── vector_store.py
            │   └── retriever.py
            └── vision/
                ├── __init__.py
                ├── config.py
                ├── models.py
                ├── constants.py
                ├── prompt.py
                ├── client.py
                ├── parser.py
                ├── interpreter.py
                ├── validation.py
                └── evaluation.py

Not every Phase 2 file necessarily exists yet.

## Core Principles

Core modules under `src/human_design/rag/` and `src/human_design/vision/` must be:

- reusable
- interface-agnostic
- local-first
- typed
- independently testable
- free of user-facing printing
- free of CLI argument parsing
- explicit about errors and side effects

Scripts may parse CLI arguments, print output, format user-facing errors, and choose process exit codes.

Core modules must not parse CLI arguments or print user-facing output.

Core modules must not import:

- `argparse`
- `streamlit`
- `rich`
- `boto3`
- AWS SDKs
- `FastAPI`
- UI code
- LangChain
- database clients
- web framework code

The only Phase 2 module allowed to import the approved Vision provider SDK is:

    src/human_design/vision/client.py

The only Phase 1 modules allowed to import LlamaIndex, Chroma, or embedding-provider code are the relevant Phase 1 implementation modules.

## Phase 1 RAG Module Responsibilities

Phase 1 remains a protected baseline.

### rag/config.py

Responsible for:

- loading Phase 1 environment variables
- defining RAG configuration
- resolving local PDF and Chroma paths
- storing chunk size and chunk overlap
- storing collection name
- storing embedding model name
- storing ingestion version

### rag/models.py

Responsible for typed Phase 1 result objects, such as:

- PDF load result
- ingestion result
- chunking result
- vector store result
- retrieval result

### rag/ingestion.py

Responsible for:

- discovering local PDFs
- loading PDF documents
- text-only extraction behavior
- attaching metadata
- chunking documents
- returning chunks or nodes for persistence

Do not put Chroma persistence logic in this module.

### rag/vector_store.py

Responsible for:

- Chroma persistent client setup
- collection setup and metadata handling
- LlamaIndex vector-store integration
- storing nodes or chunks
- opening existing Chroma-backed collections safely

### rag/retriever.py

Responsible for:

- loading the existing vector store
- running top-k retrieval
- returning typed retrieval results
- preserving source metadata

## Phase 2 Vision Module Responsibilities

### vision/config.py

Responsible for:

- loading Phase 2 environment variables
- defining `VisionConfig` or equivalent typed configuration
- resolving local BodyGraph sample paths
- storing Vision model configuration
- enforcing safe defaults such as `HD_VISION_REAL_API=0`
- checking whether explicit real API use is enabled

Do not mix Phase 1 RAG configuration into this module unless a setting is truly shared.

Do not hardcode environment-dependent values throughout the codebase.

### vision/models.py

Responsible for typed schemas and result objects, including:

- planetary activation values
- Personality and Design activation columns
- raw Vision extraction
- confidence and uncertainty data
- parser result and parser issues
- derived basic information
- derived chart data
- validation warnings or issues
- validation result
- evaluation result
- full BodyGraph extraction result

Models must not import Vision clients.

Use dataclasses or typed schema models consistently.

If a function returns more than two related values, prefer a typed result object over a loose tuple or dictionary.

### vision/constants.py

Responsible for stable deterministic domain constants only.

This includes:

- canonical center names
- parser-only center aliases
- canonical center ordering
- exactly 36 canonical Human Design channels
- channel-to-center mapping
- motor-center definitions
- other fixed deterministic domain mappings

Do not place parsing logic, Vision API logic, validation logic, or interpretation logic in this module.

Do not duplicate canonical channel or center mappings in other modules.

### vision/prompt.py

Responsible for:

- loading the raw Vision extraction prompt from `prompts/`
- formatting prompt content if needed
- exposing prompt text through a small typed or string-returning API

The prompt must request raw visible facts only.

Do not derive type, authority, profile, definition, strategy, not-self theme, or signature in this module.

Do not place API calls in this module.

### vision/client.py

Responsible for:

- approved Vision provider setup
- local image input preparation
- explicit real API opt-in enforcement
- sending the raw extraction prompt and image to the provider
- returning raw provider response data to the caller

This module must:

- require explicit opt-in before a real API call
- never print or log API keys
- never print or log full base64 image content
- never write raw private responses to disk by default
- keep provider-specific request and response details isolated here

This module must not:

- parse final JSON into domain objects
- derive chart concepts
- validate Human Design rules
- generate readings
- import RAG retrieval modules

### vision/parser.py

Responsible for:

- parsing strict JSON returned by the Vision provider
- normalizing planetary activation values
- normalizing center aliases
- normalizing valid channel orientation and whitespace
- creating typed raw extraction objects
- preserving recoverable parser issues for validation
- raising clear exceptions for invalid JSON or missing required top-level structure

Parser behavior must preserve the raw-versus-derived boundary.

The parser must not:

- call a Vision API
- call an LLM
- derive active gates
- derive active channels
- derive defined centers
- derive type, authority, profile, definition, strategy, not-self theme, or signature

### vision/interpreter.py

Responsible for deterministic Python derivation only.

This includes:

- deriving active gates from Personality and Design planetary activations
- deriving active channels from canonical channels
- deriving defined centers from active-channel endpoints
- deriving profile from Personality Sun and Design Sun lines
- deriving definition from graph connectivity
- deriving type from deterministic center and channel rules
- deriving authority conservatively
- deriving strategy, not-self theme, and signature from type
- returning conservative `Unknown` or `Needs Review` outcomes for unsupported cases

The interpreter must not:

- call a Vision API
- call an LLM
- read image files
- use visual evidence as direct deterministic truth
- import CLI code
- import RAG retrieval code

### vision/validation.py

Responsible for comparing:

- parser issues
- raw visual evidence
- deterministic derived chart data

Validation must produce typed machine-readable warnings or issues.

Validation may:

- merge parser warnings into final validation output
- compare visual gates against derived active gates
- compare visible channels against derived active channels
- compare visual centers against derived centers
- detect unsupported authority cases
- determine `is_valid` from issue severity and validity effects

Validation must not:

- call a Vision API
- call an LLM
- silently rewrite derived chart data
- reimplement the interpreter's deterministic derivation logic

### vision/evaluation.py

Responsible for:

- loading or accepting golden labels and predictions
- comparing raw extraction output with raw golden labels
- comparing deterministic derived output with derived golden labels
- calculating exact-match metrics
- calculating set-based precision, recall, and F1 metrics
- calculating aggregate macro metrics
- evaluating warning-code behavior
- exposing threshold checks through typed results or exceptions

Evaluation must not:

- call a Vision API by default
- require private images
- require API keys
- generate readings
- contain CLI argument parsing

### scripts/extract_bodygraph.py

Responsible for:

- parsing CLI arguments
- loading configuration
- choosing mock-response or real API mode
- calling Vision client, parser, interpreter, and validation modules
- formatting safe JSON or text output
- converting expected exceptions into clear user-facing CLI errors
- choosing exit codes

This script must not contain core parser, interpreter, validation, or evaluation logic.

### scripts/evaluate_bodygraph_extraction.py

Responsible for:

- parsing local evaluation CLI arguments
- loading golden labels and saved predictions
- calling evaluation functions
- printing per-case and aggregate metrics
- returning an appropriate process exit code when thresholds fail

This script must not contain evaluation metric implementation details.

## Path Handling

Use `pathlib.Path` for filesystem paths inside core modules.

Good example:

    def load_prompt(prompt_path: Path) -> str:
        ...

Avoid:

    def load_prompt(prompt_path: str) -> str:
        ...

Use `Path` for:

- image paths
- prompt paths
- JSON fixture paths
- local sample directories
- configuration file paths
- golden-label paths
- prediction paths

## Type Hints

Use type hints on all public functions.

Use Python 3.10+ union syntax.

Good example:

    def get_vision_config(env_file: Path | None = None) -> VisionConfig:
        ...

Avoid untyped public functions.

Use precise collection types where helpful.

Good example:

    def derive_active_channels(active_gates: set[int]) -> tuple[str, ...]:
        ...

Avoid using `dict[str, object]` or `Any` for core domain results when a typed model can represent the data.

## Error Handling Boundary

Core functions should raise clear exceptions or return typed result objects.

Scripts should catch exceptions and format user-facing messages.

Core example:

    def load_prompt(prompt_path: Path) -> str:
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

Script example:

    try:
        prompt = load_prompt(prompt_path)
    except FileNotFoundError as exc:
        print(f"Cannot run BodyGraph extraction: {exc}")
        raise SystemExit(1)

Use hard exceptions for:

- missing local files required to run a task
- invalid JSON
- missing required top-level schema structure
- invalid configuration
- real API usage requested without explicit opt-in
- real API usage requested without an API key

Use structured warnings or issues for:

- recoverable Vision uncertainty
- reversed valid channel normalization
- invalid visual evidence
- missing visual evidence
- visual-versus-derived disagreement
- unsupported authority outcomes

## Public and Private Helpers

Public APIs should be clear and task-oriented.

Examples:

    def parse_raw_vision_response(raw_json: str) -> ParseResult:
        ...

    def derive_chart_data(raw_vision: RawVisionExtraction) -> DerivedChartData:
        ...

    def validate_bodygraph(
        parse_result: ParseResult,
        derived_chart_data: DerivedChartData,
    ) -> ValidationResult:
        ...

Private helpers should start with an underscore.

Examples:

    def _normalize_channel(value: str) -> tuple[str | None, ValidationWarning | None]:
        ...

    def _derive_active_gates(
        personality: ActivationColumn,
        design: ActivationColumn,
    ) -> set[int]:
        ...

## Deterministic Data Flow

Phase 2 core logic must follow this direction:

    Raw provider JSON
       ->
    Parser
       ->
    Typed raw Vision extraction plus parser issues
       ->
    Deterministic interpreter
       ->
    Derived chart data
       ->
    Validation
       ->
    Full BodyGraph extraction result

Do not reverse the direction.

Visual evidence must not overwrite:

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

## Testability Rules

Default tests must remain offline.

Use:

- raw JSON fixtures
- in-memory typed models
- mocked Vision client responses
- temporary local files where needed
- synthetic or non-private fixture images only

Do not require:

- OpenAI API keys
- real Vision calls
- private BodyGraph images
- Phase 1 Chroma storage
- PDF ingestion
- network access

Inject or monkeypatch Vision client behavior at the script or client boundary rather than mocking parser, interpreter, or validation behavior unnecessarily.

## File and Function Size

- Module greater than 300 lines: consider splitting it.
- Function greater than 50 lines: consider extracting helpers.
- A long deterministic rule table may remain in `constants.py` when splitting it would duplicate or obscure the canonical source of truth.

## Dependencies

Do not add future-phase dependencies.

During Phase 2, do not add:

- `boto3`
- `streamlit`
- `fastapi`
- `langchain`
- `pinecone`
- `sqlalchemy`
- `llama-parse`
- OCR packages
- training frameworks
- YOLO dependencies
- annotation UI libraries
- cloud SDKs

Only add a dependency when the active Phase 2 task explicitly requires it and the dependency is consistent with `AGENTS.md` and `docs/phase2_implementation_plan.md`.