# BodyGraph extraction

This replaces the completed Phase 2 task plan with the current chart contract. Normal CLI and Streamlit use is in the [README](../README.md).

## Pipeline

```text
Image -> Vision raw extraction -> strict parser
  -> deterministic interpreter -> validation -> BodyGraphExtractionResult
```

- `vision/client.py` owns image preparation and the OpenAI call. Real extraction requires `HD_VISION_REAL_API=1` and `OPENAI_API_KEY`.
- `HD_VISION_MODEL` and `HD_VISION_REASONING_EFFORT` select image-extraction settings; they are separate from the answer model.
- `vision/parser.py` parses the provider JSON, normalizes accepted aliases and records recoverable issues.
- `vision/interpreter.py` derives chart facts using the canonical rules in `vision/constants.py`.
- `vision/validation.py` combines parser issues and visual/deterministic disagreements into typed warnings and validity.
- `vision/pipeline.py` returns the existing frozen `BodyGraphExtractionResult` containing raw extraction, derived chart data and validation.

## Chart truth

The 13 Personality and 13 Design planetary activations determine active Gates. Active Channels require both canonical endpoint Gates; defined Centers come from those Channels. Python rules derive Type, Authority, Profile, Strategy, Definition, not-self theme and signature.

Visual-only Gates, Channels and Centers are supporting evidence and cannot override those derived facts. Canonical Center vocabulary includes `G` and `Ego`. Unsupported or incomplete information is reported conservatively, not invented.

The reading pipeline rejects invalid charts before retrieval. Its adapter exposes only relevant deterministic facts and minimal safe warnings. Names, birth details, images, confidence, uncertain items and raw provider payloads do not enter the answer prompt as chart metadata.

## Standalone tools

- `scripts/extract_bodygraph.py` extracts one local image; `--json` emits structured output. `--mock-response` supports sanitized offline fixtures.
- `scripts/evaluate_bodygraph_extraction.py` compares saved predictions against manually verified labels for the same chart/case ID. It does not call Vision.
- Golden labels use `phase2_golden_labels_v2`; wrapped predictions use `phase2_predictions_v1`. See `data/bodygraph_samples/golden_labels.example.json` for a synthetic example.
- Evaluation reports activation exact matches, set precision/recall/F1, basic-info matches and warning metrics. Missing applicable predictions count as failures; malformed supplied data fails validation. Optional aggregate thresholds can fail evaluation.

Use `--help` for tool arguments. Keep private images, predictions and labels under the ignored `data/bodygraph_samples/` directories. Do not compare a real chart against synthetic example labels.

## Regression coverage

The parser, interpreter, validation and evaluation tests use typed data, sanitized fixtures and fake clients. They require neither private images nor paid services. `test_extract_bodygraph.py` verifies the integrated extraction facade. Run the [offline verification](../README.md#default-verification).
