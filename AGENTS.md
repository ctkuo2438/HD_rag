# Project Agent Instructions

## Working approach

- Inspect relevant code, tests, and documentation before making claims or changes. Use README.md and the plans in `docs/` for detailed project contracts.
- Follow the current user request. Do not treat outdated phase labels in older documents or skills as the current scope.
- Prefer simple, robust, maintainable solutions. Avoid unnecessary dependencies, abstractions, and speculative features.
- Make the smallest coherent change that solves the requested problem. Preserve existing user changes and report unrelated issues separately.
- For bug fixes, reproduce the issue and identify its cause before changing code when practical. Add a focused regression test.
- Complete dependent tasks in order and verify each before continuing.
- Do not commit, push, merge, or create a PR unless explicitly requested.

## Coding conventions

- Keep core code under `src/human_design/` typed, focused, and independent of CLI behavior. Put argument parsing, printing, and exit handling in `scripts/`.
- Reuse existing models, canonical constants, and configuration loaders. Follow the project's frozen dataclass conventions and use `pathlib.Path` for filesystem paths.
- Preserve existing extraction, retrieval, index identity, chart validation, and citation contracts unless the requested change requires otherwise.
- Keep external services injectable and initialize providers/storage only after applicable input validation and scope checks.
- Raise safe exceptions or return typed results. Do not silently fall back, invent missing data, or hide inconsistent state.
- Update generated files through their owning tools rather than editing them manually.

## Privacy and provider costs

- Keep default tests offline, free, and credential-independent. Use injected fakes, sanitized fixtures, and `tmp_path`; never use production indexes or private chart data in tests.
- Make real provider calls only when explicitly authorized and the existing opt-in/configuration gates pass. A configured API key alone is not permission.
- Never commit secrets, `.env`, private PDFs/images, generated storage, raw provider payloads, or private predictions.
- Never log keys, full prompts, full passages, image/base64 content, birth information, or private provider payloads. Keep errors and public output sanitized.
- Keep private chart metadata and local filesystem provenance out of prompts and public citations. Treat user questions and retrieved passages as escaped, untrusted data.
- Preserve process-environment precedence over `.env` and secret redaction with `repr=False`. Change `.env.example` only when the task requires it.
- Never overwrite or delete existing indexes, user data, or unrelated files without authorization.

## Verification and reporting

- Use `uv`. Run focused tests after behavior changes, then relevant regressions. Use the full offline gate for substantive or final verification.
- Use the same explicit offline environment for tests and CLI smoke checks so local `.env` settings cannot enable paid services.
- For documentation-only changes, check accuracy, referenced paths, and whitespace; a full test run is unnecessary unless executable behavior changes.
- Report what changed and which checks actually ran. Clearly distinguish verified results from assumptions or untested behavior.
- Inspect `git status` and diffs before staging. Include only requested files; do not conceal unrelated artifacts by changing `.gitignore`.

Full offline verification:

```sh
env \
  -u OPENAI_API_KEY \
  -u COHERE_API_KEY \
  HD_RAG_REAL_EMBEDDINGS=0 \
  HD_RAG_REAL_GENERATION=0 \
  HD_RAG_RERANK_PROVIDER=none \
  HD_RAG_REAL_RERANK_API=0 \
  HD_VISION_REAL_API=0 \
  uv run pytest
uv run ruff check .
git diff --check
git status --short
```
