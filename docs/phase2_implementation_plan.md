# Phase 2 Implementation Plan: Local BodyGraph Vision Extraction

> **For agentic workers:** Implement this plan task-by-task. Use TDD where practical: write each test first, verify it fails for the expected reason, implement the minimum code, then verify it passes. Use available planning/execution skills when helpful.

**Goal:** Build a local Vision extraction pipeline for Human Design BodyGraph / chart images.

**Architecture:** Phase 2 takes a local BodyGraph image, asks a Vision model to extract only raw visible facts, parses strict JSON, derives chart data with deterministic Python rules, validates consistency, and supports evaluation plus local CLI smoke testing. Phase 2 does not generate Human Design readings.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, existing dotenv configuration approach, one Vision API provider, and optional schema/image helpers if they are truly needed.

---

## Goal

Build a local Vision extraction pipeline for Human Design BodyGraph / chart images.

Phase 2 should convert a local BodyGraph image into structured chart data that can later be used by the RAG reading system.

Phase 2 is extraction, deterministic interpretation, validation, evaluation, and local CLI smoke testing only. It is not a reading generator and does not perform RAG answer generation.

## Core Architecture Decision

Do not ask the Vision model to directly infer final Human Design concepts such as:

- type
- authority
- profile
- definition
- strategy
- not_self_theme
- signature

Instead, split the pipeline into:

1. Vision extraction: extract raw visible facts from the image.
2. Parser: parse and normalize strict JSON.
3. Deterministic interpreter: derive Human Design concepts using Python rules.
4. Validation: check consistency between raw facts, derived facts, and visual evidence.
5. Evaluation: compare predictions against golden labels.

The reason for this architecture is that Vision models may hallucinate type, authority, channels, or Reflector charts. Deterministic Python rules make the pipeline testable and reduce hallucination risk.

## Non-Goals

Explicitly out of scope for Phase 2:

- AWS
- S3
- Lambda
- API Gateway
- EC2
- SageMaker
- Streamlit
- FastAPI
- web app
- user accounts
- RAG answer generation
- Claude / GPT reading generation
- Chroma ingestion
- PDF ingestion
- modifying Phase 1 retrieval architecture
- OCR for arbitrary PDFs
- training computer vision models
- YOLO / object detection training
- manual annotation UI
- cloud deployment
- production API
- frontend
- database persistence
- astrology calculation from birth date/time/place

## Architecture

Use this fixed Phase 2 pipeline:

```text
Local BodyGraph image
   ↓
Vision API raw extraction prompt
   ↓
Strict JSON parser
   ↓
Raw BodyGraph facts + uncertainty/confidence
   ↓
Deterministic Python interpreter
   ↓
Derived chart_data.basic_info
   ↓
Validation rules and warnings
   ↓
Evaluation script + CLI smoke test
```

Do not change this architecture unless the user explicitly changes Phase 2 scope.

## Tech Stack

- Python 3.11+: runtime for the local Vision extraction pipeline.
- uv: dependency management, virtual environment, and command runner.
- pytest: default test runner.
- ruff: linting.
- python-dotenv: configuration loading if reused from the Phase 1 pattern.
- OpenAI Python SDK: preferred Vision provider if a provider dependency is needed, because Phase 1 already uses OpenAI embeddings.
- pydantic: optional, only if useful for schema validation.
- pillow: optional, only if basic local image validation or metadata checks are needed.

Prefer one Vision provider for Phase 2 to reduce complexity. The default provider recommendation is OpenAI Vision API. Only keep provider abstraction if the repo clearly needs it; otherwise avoid dual OpenAI/Anthropic abstraction in Phase 2.

Do not add web frameworks, AWS packages, Streamlit, FastAPI, database dependencies, deployment dependencies, OCR packages, or computer vision training dependencies.

## Target File Map

```text
<repo-root>/
├── data/
│   └── bodygraph_samples/
│       ├── images/                 # ignored by git by default
│       ├── private/                # ignored by git
│       └── golden_labels.example.json
├── prompts/
│   └── bodygraph_raw_extraction.txt
├── scripts/
│   ├── extract_bodygraph.py
│   └── evaluate_bodygraph_extraction.py
├── src/
│   └── human_design/
│       └── vision/
│           ├── __init__.py
│           ├── config.py
│           ├── models.py
│           ├── constants.py
│           ├── prompt.py
│           ├── client.py
│           ├── parser.py
│           ├── interpreter.py
│           ├── validation.py
│           └── evaluation.py
├── tests/
│   ├── fixtures/
│   │   └── bodygraph/
│   │       ├── test1.png
│   │       ├── test1_raw_response.json
│   │       └── predictions.example.json
│   ├── test_vision_config.py
│   ├── test_vision_models.py
│   ├── test_bodygraph_constants.py
│   ├── test_bodygraph_parser.py
│   ├── test_bodygraph_interpreter.py
│   ├── test_bodygraph_validation.py
│   ├── test_bodygraph_evaluation.py
│   └── test_extract_bodygraph.py
├── docs/
│   └── phase2_implementation_plan.md
```

Private BodyGraph images should not be committed by default. Unit tests should use mocked JSON responses and sanitized fixtures. `tests/fixtures/bodygraph/test1.png` should be a small committed non-private or synthetic BodyGraph-like image used only for mock/offline CLI smoke tests. It must not contain private user chart data. Real/private chart images remain local-only under `data/bodygraph_samples/images/` or `data/bodygraph_samples/private/` and should be ignored by git.

## Required Schema Design

The output schema must separate:

- raw extraction
- derived chart data
- validation

Raw extraction must include confidence or uncertainty fields.

Conceptual schema:

```json
{
  "raw_vision": {
    "personality": {
      "sun": "61.4",
      "earth": "62.4",
      "north_node": "3.1",
      "south_node": "60.2",
      "moon": "10.3",
      "mercury": "34.2",
      "venus": "30.1",
      "mars": "41.5",
      "jupiter": "50.2",
      "saturn": "59.2",
      "uranus": "18.1",
      "neptune": "22.2",
      "pluto": "14.3"
    },
    "design": {
      "sun": "32.6",
      "earth": "42.6",
      "north_node": "5.1",
      "south_node": "44.1",
      "moon": "60.3",
      "mercury": "3.4",
      "venus": "34.1",
      "mars": "10.4",
      "jupiter": "61.1",
      "saturn": "62.1",
      "uranus": "41.1",
      "neptune": "14.1",
      "pluto": "50.1"
    },
    "visually_defined_centers": [],
    "visually_undefined_centers": [],
    "visually_active_gates": [],
    "visible_colored_channels": [],
    "confidence": {
      "personality": {
        "sun": 0.0,
        "earth": 0.0,
        "north_node": 0.0,
        "south_node": 0.0,
        "moon": 0.0,
        "mercury": 0.0,
        "venus": 0.0,
        "mars": 0.0,
        "jupiter": 0.0,
        "saturn": 0.0,
        "uranus": 0.0,
        "neptune": 0.0,
        "pluto": 0.0
      },
      "design": {
        "sun": 0.0,
        "earth": 0.0,
        "north_node": 0.0,
        "south_node": 0.0,
        "moon": 0.0,
        "mercury": 0.0,
        "venus": 0.0,
        "mars": 0.0,
        "jupiter": 0.0,
        "saturn": 0.0,
        "uranus": 0.0,
        "neptune": 0.0,
        "pluto": 0.0
      },
      "visually_defined_centers": 0.0,
      "visually_undefined_centers": 0.0,
      "visually_active_gates": 0.0,
      "visible_colored_channels": 0.0
    },
    "uncertain_items": [
      {
        "field_path": "visible_colored_channels[0]",
        "observed_value": "10-57",
        "reason": "partially obscured or visually ambiguous color",
        "confidence": 0.45
      }
    ]
  },
  "derived_chart_data": {
    "basic_info": {
      "type": "",
      "authority": "",
      "profile": "",
      "strategy": "",
      "definition": "",
      "not_self_theme": "",
      "signature": ""
    },
    "active_gates": [],
    "active_channels": [],
    "defined_centers": []
  },
  "validation": {
    "is_valid": true,
    "warnings": [
      {
        "code": "VISIBLE_CHANNEL_NOT_DERIVED",
        "message": "Visible channel was not derived from planetary activations.",
        "severity": "WARNING",
        "affects_validity": false,
        "source": "validation"
      }
    ]
  }
}
```

A normal activation value is a `Gate.Line` string such as `"61.4"`. JSON `null` is the canonical representation when a required activation cannot be read. Empty strings are not the canonical representation for unavailable activations and are normalized as unavailable values during parsing.

Important distinction:

- `visually_defined_centers`: centers the Vision model believes appear colored or defined in the image.
- `visually_undefined_centers`: centers the Vision model believes appear white or undefined in the image.
- `visually_active_gates`: gates the Vision model sees as colored on the BodyGraph.
- `active_gates`: gates deterministically derived from Personality and Design planetary activations.
- `visible_colored_channels`: channels the Vision model sees as fully colored.
- `active_channels`: channels deterministically derived from `active_gates`.
- `defined_centers`: centers deterministically derived from endpoints of column-derived active channels.

Type, authority, and definition must use `derived_chart_data.defined_centers`. Visual center detection is supporting evidence for validation only and must not directly determine type, authority, definition, strategy, not_self_theme, or signature.

The exact Python model class names can be adjusted, but this conceptual separation must remain.

### Canonical Activation Fields

All 26 activation keys are required in the normalized raw Vision schema.

Personality fields:

```text
sun
earth
north_node
south_node
moon
mercury
venus
mars
jupiter
saturn
uranus
neptune
pluto
```

Design fields:

```text
sun
earth
north_node
south_node
moon
mercury
venus
mars
jupiter
saturn
uranus
neptune
pluto
```

The Vision prompt should request every canonical activation key. `null` is the canonical raw JSON representation when the Vision model cannot read a required activation. Blank strings should be treated equivalently to unavailable values.

Parser normalization should preserve unavailable activation values as `None` or an equivalent typed nullable representation. Missing, blank, or null individual activation values are recoverable extraction-quality failures that reach validation as structured issues. Missing `personality` or `design` containers remain hard parser errors because they are required schema structure.

Missing activation issue rules:

- missing `personality.sun` produces `MISSING_PERSONALITY_SUN`, not both `MISSING_PERSONALITY_SUN` and `MISSING_ACTIVATION`.
- missing `design.sun` produces `MISSING_DESIGN_SUN`, not both `MISSING_DESIGN_SUN` and `MISSING_ACTIVATION`.
- missing any other required Personality or Design planetary field produces `MISSING_ACTIVATION`.
- malformed non-empty values such as `"unknown"` or `"61.x"` produce `MALFORMED_ACTIVATION`, not `MISSING_ACTIVATION`.
- out-of-range gates and lines produce `INVALID_ACTIVATION_GATE` or `INVALID_ACTIVATION_LINE`.
- missing or invalid required activation data makes `validation.is_valid` false because deterministic active gates, channels, centers, and final chart interpretation could otherwise be incomplete.
- unavailable activation values must not be silently treated as empty gates, false values, or absent active gates.
- confidence and `uncertain_items` are metadata only and must not bypass missing-activation validation.

### Confidence and Uncertainty

Confidence values must be numeric only. Every confidence value must be a finite float in the inclusive range:

```text
0.0 <= confidence <= 1.0
```

Do not use percentages, strings such as `"high"` or `"low"`, or free-form confidence text.

`uncertain_items` records ambiguous raw visual observations. Rules:

- `field_path` must point only to a raw Vision field, never a derived chart field.
- `observed_value` records what the Vision model actually observed before deterministic derivation.
- `reason` is a concise explanation of the visual ambiguity.
- `confidence` uses the same numeric `0.0` to `1.0` range.
- Empty `uncertain_items` is valid.
- Confidence and uncertainty are raw extraction metadata, not direct inputs to deterministic interpretation.
- Phase 2 evaluation does not need to score confidence calibration yet.
- The parser must reject or clearly flag confidence values outside the allowed range.
- Do not add a confidence threshold that changes chart derivation behavior in Phase 2.

### Validation Issue Shape

Every structured validation warning or issue must include at least:

- `code`: machine-readable string
- `message`: human-readable string
- `severity`: one of `INFO`, `WARNING`, or `ERROR`
- `affects_validity`: boolean
- `source`: one of `parser`, `interpreter`, or `validation`

`validation.is_valid` is derived from warning validity effects:

```text
validation.is_valid == not any(issue.affects_validity for issue in validation.warnings)
```

`is_valid` measures whether the structured chart result is sufficiently valid for deterministic derivation. It is not a score of Vision quality and does not become false solely because visual evidence differs from derived chart data.

### Canonical Center Vocabulary

After parsing, interpreter, validation, evaluation, and golden-label logic must use exactly these nine canonical center names:

```text
Head
Ajna
Throat
G
Ego
Sacral
Spleen
Solar Plexus
Root
```

Parser-only aliases should normalize to canonical center names:

```text
Heart / Will / Ego -> Ego
G Center / Self / Identity -> G
Emotional / Solar Plexus -> Solar Plexus
```

Raw Vision responses may contain aliases, but parsed models and downstream logic should use only canonical center names.

## Raw Facts Source Of Truth

Active gates are derived only from the Personality and Design activation columns.

Rules:

- Parser normalizes planetary activation values like `61.4`.
- Interpreter derives gate numbers from all Personality and Design planetary activations.
- Example: `61.4` contributes gate `61`.
- The union of all Personality and Design gate numbers becomes `active_gates`.
- Active channels are derived only from `active_gates` using canonical `ALL_CHANNELS`.
- `derived_chart_data.defined_centers` are derived from endpoints of `active_channels`.
- Type, authority, definition, strategy, not_self_theme, and signature use derived chart data.
- Visual gates, visible colored channels, and visually defined centers are additional evidence only.
- Vision-reported facts must not directly become deterministic derived facts.
- Validation warns if column-derived active gates disagree with visually detected active gates.
- Validation warns if visible colored channels are not present in derived active channels.
- Validation warns if visually defined centers disagree with derived centers.

Example:

```text
Column-derived active channels:
3-60, 10-34

Derived defined centers:
Root, Sacral, G

Vision-reported centers:
Root, Sacral, G, Throat

Validation result:
warning that visually defined Throat is not explained by derived active channels
```

### Planetary Activation Format

Planetary values in the Personality and Design columns use `Gate.Line` format.

Examples:

```text
60.1 = Gate 60, Line 1
50.4 = Gate 50, Line 4
```

Gate values must be integers from 1 to 64. Line values must be integers from 1 to 6.

The parser should convert each activation into a typed object, for example:

```text
Activation(gate=60, line=1)
Activation(gate=50, line=4)
```

The line is used for profile derivation:

```text
profile = Personality Sun line / Design Sun line
```

Example:

```text
Personality Sun = 60.1
Design Sun = 50.4
profile = 1/4
```

### Parser Errors Versus Validation Warnings

Invalid JSON and missing required structural fields are hard parser errors. Imperfect Vision extraction should generally survive parsing and reach validation so evaluation can measure the failure.

| Situation | Responsible module | Required outcome |
|---|---|---|
| Invalid JSON | Parser | Raise a clear parse error |
| Missing required top-level schema fields | Parser | Raise a clear parse error |
| Missing `personality` or `design` container | Parser | Raise a clear parse error |
| Valid activation such as `61.4` | Parser | Normalize into typed activation |
| Null activation value | Parser + Validation | Preserve as unavailable and emit the appropriate missing-activation issue |
| Blank activation string | Parser + Validation | Preserve as unavailable and emit the appropriate missing-activation issue |
| Omitted individual activation key inside otherwise valid `personality` or `design` data | Parser + Validation | Normalize as unavailable and emit the appropriate missing-activation issue |
| Missing `personality.sun` | Validation | Emit `MISSING_PERSONALITY_SUN` only |
| Missing `design.sun` | Validation | Emit `MISSING_DESIGN_SUN` only |
| Missing non-Sun Personality or Design activation | Validation | Emit `MISSING_ACTIVATION` |
| Malformed non-empty activation such as `unknown` or `61.x` | Parser + Validation | Preserve the issue and emit `MALFORMED_ACTIVATION` |
| Reversed valid channel such as `57-10` | Parser | Normalize to `10-57` and preserve `VISIBLE_CHANNEL_NORMALIZED` warning information |
| Whitespace variant such as `3 - 60` | Parser | Normalize to `3-60` |
| Impossible visible channel such as `34-99` | Parser + Validation | Preserve the issue and emit `INVALID_VISIBLE_CHANNEL` issue |
| Invalid activation line such as `60.9` | Parser + Validation | Preserve the issue and emit `INVALID_ACTIVATION_LINE` issue |
| Invalid activation gate such as `99.1` | Parser + Validation | Preserve the issue and emit `INVALID_ACTIVATION_GATE` issue |
| Visual evidence disagrees with deterministic derivation | Validation | Emit machine-readable cross-source warning |

Parser normalization warnings are internal pipeline data. The model layer should support preserving them, for example with a typed result shaped like:

```text
ParseResult(
    raw_vision=RawVisionExtraction,
    warnings=tuple[ValidationWarning, ...],
)
```

The final class name may differ, but parser warnings must be preservable and mergeable into final validation output. They do not need their own top-level CLI output field.

## Human Design Derivation Rules

The implementation must use deterministic Python rules for at least the fields below.

### Profile

Derive from Sun line values:

```text
profile = personality_sun_line / design_sun_line
```

Example:

```text
Personality Sun = 61.4
Design Sun = 32.6
profile = 4/6
```

### Active Gates

Derive from the gate numbers in all Personality and Design planetary activation values.

Example:

```text
Personality Sun = 61.4
Design Sun = 32.6
Personality Earth = 62.4
Design Earth = 42.6

active_gates includes 61, 32, 62, 42
```

### Active Channels

Use a canonical fixed `ALL_CHANNELS` mapping for all 36 Human Design channels.

A channel is active if both gates are active.

Example:

```text
active_gates includes 3 and 60
=> active_channels includes 3-60
```

`ALL_CHANNELS` must contain exactly these 36 canonical channels:

```text
1-8
2-14
3-60
4-63
5-15
6-59
7-31
9-52
10-20
10-34
10-57
11-56
12-22
13-33
16-48
17-62
18-58
19-49
20-34
20-57
21-45
23-43
24-61
25-51
26-44
27-50
28-38
29-46
30-41
32-54
34-57
35-36
37-40
39-55
42-53
47-64
```

Channel strings should be normalized to canonical orientation. For example, `57-10` should normalize to `10-57`, and `3 - 60` should normalize to `3-60`.

A channel that is not in `ALL_CHANNELS` is invalid and should produce a validation warning or parser error, depending on context.

Vision-reported `visible_colored_channels` are raw visual evidence, not final truth. The interpreter must derive `active_channels` only by iterating over `ALL_CHANNELS`; a channel is active only when both gates are present in the column-derived `active_gates`. Derived `active_channels` must never include channels outside `ALL_CHANNELS`. Vision-reported `visible_colored_channels` must not directly become derived `active_channels`; they are used only as supporting evidence for validation.

Conceptual example:

```text
active_gates = {3, 10, 34, 57, 60}

ALL_CHANNELS contains:
3-60
10-34
10-57
34-57

derived active_channels:
3-60
10-34
10-57
34-57
```

#### Canonical Channel-to-Center Mapping

This table is the single normative source for `CHANNEL_TO_CENTERS` in Phase 2.

| Channel | Centers |
|---|---|
| `1-8` | `G`, `Throat` |
| `2-14` | `G`, `Sacral` |
| `3-60` | `Sacral`, `Root` |
| `4-63` | `Ajna`, `Head` |
| `5-15` | `Sacral`, `G` |
| `6-59` | `Solar Plexus`, `Sacral` |
| `7-31` | `G`, `Throat` |
| `9-52` | `Sacral`, `Root` |
| `10-20` | `G`, `Throat` |
| `10-34` | `G`, `Sacral` |
| `10-57` | `G`, `Spleen` |
| `11-56` | `Ajna`, `Throat` |
| `12-22` | `Throat`, `Solar Plexus` |
| `13-33` | `G`, `Throat` |
| `16-48` | `Throat`, `Spleen` |
| `17-62` | `Ajna`, `Throat` |
| `18-58` | `Spleen`, `Root` |
| `19-49` | `Root`, `Solar Plexus` |
| `20-34` | `Throat`, `Sacral` |
| `20-57` | `Throat`, `Spleen` |
| `21-45` | `Ego`, `Throat` |
| `23-43` | `Throat`, `Ajna` |
| `24-61` | `Ajna`, `Head` |
| `25-51` | `G`, `Ego` |
| `26-44` | `Ego`, `Spleen` |
| `27-50` | `Sacral`, `Spleen` |
| `28-38` | `Spleen`, `Root` |
| `29-46` | `Sacral`, `G` |
| `30-41` | `Solar Plexus`, `Root` |
| `32-54` | `Spleen`, `Root` |
| `34-57` | `Sacral`, `Spleen` |
| `35-36` | `Throat`, `Solar Plexus` |
| `37-40` | `Solar Plexus`, `Ego` |
| `39-55` | `Root`, `Solar Plexus` |
| `42-53` | `Sacral`, `Root` |
| `47-64` | `Ajna`, `Head` |

Requirements:

- `CHANNEL_TO_CENTERS` keys must exactly equal `ALL_CHANNELS`.
- Every channel maps to exactly two distinct canonical center names.
- No alternate or duplicate mapping may exist elsewhere in the implementation.
- Derived defined centers are the set union of all endpoint centers from derived active channels.
- Output center lists must use the existing canonical center order:

```text
Head
Ajna
Throat
G
Ego
Sacral
Spleen
Solar Plexus
Root
```

The example case should continue to output defined centers as:

```json
["G", "Sacral", "Solar Plexus", "Root"]
```

### Definition

Derive `derived_chart_data.defined_centers` from the center endpoints of deterministic `active_channels`. Treat derived centers as graph nodes and active channels as graph edges between centers.

Interpreter invariant:

```text
derived_chart_data.defined_centers must equal the canonical-order union of all endpoint centers from derived active_channels.
```

Examples:

```text
active_channels = []
-> defined_centers = []

active_channels = ["3-60"]
-> defined_centers = ["Sacral", "Root"]

active_channels = ["3-60", "10-34", "30-41"]
-> defined_centers = ["G", "Sacral", "Solar Plexus", "Root"]
```

Use connected components to derive:

```text
0 components -> No Definition
1 component -> Single Definition
2 components -> Split Definition
3 components -> Triple Split Definition
4 components -> Quadruple Split Definition
```

Phase 2 v1 intentionally uses this coarse connected-component taxonomy. `Split Definition` is the Phase 2 v1 output for any chart with exactly two disconnected derived components. Phase 2 v1 does not distinguish `Simple-Split Definition` from `Wide-Split Definition`; that finer distinction depends on bridge-specific analysis and is intentionally out of scope for this phase. The current connected-component output is a coarse deterministic classification, not a complete reproduction of every Human Design definition subtype.

### Type

Use deterministic rules based on `derived_chart_data.defined_centers` and `active_channels`.

At minimum:

In this table, `defined` means present in `derived_chart_data.defined_centers`.

```text
No derived centers -> Reflector
Sacral defined + motor-to-Throat connection -> Manifesting Generator
Sacral defined + no motor-to-Throat connection -> Generator
Sacral undefined + motor-to-Throat connection -> Manifestor
Sacral undefined + at least one defined center + no motor-to-Throat connection -> Projector
```

Motor centers:

- Root
- Sacral
- Solar Plexus
- Ego

For Phase 2 v1, a motor-to-Throat connection exists only when at least one directly active canonical channel has `Throat` as one endpoint and one motor center (`Root`, `Sacral`, `Solar Plexus`, or `Ego`) as the other endpoint. Graph connectivity may still be used for Definition, but a multi-hop path through intermediate centers must not by itself qualify as a motor-to-Throat connection for Type classification. A Sacral-defined chart is a Manifesting Generator only when it has this direct active motor-to-Throat channel; without one, it is a Generator. A Sacral-undefined chart with such a direct channel is a Manifestor, subject to the remaining Type rules above.

### Authority

Use deterministic priority rules against `derived_chart_data.defined_centers`, but handle subtle cases conservatively.

At minimum:

In this table, `defined` means present in `derived_chart_data.defined_centers`.

```text
Solar Plexus defined -> Emotional / Solar Plexus
Else Sacral defined -> Sacral
Else Spleen defined -> Splenic
Else Heart/Ego authority where explicitly supported by tested rules
Else G/Self to Throat where explicitly supported by tested rules -> Self-Projected
Else mental Projector case where explicitly supported by tested rules -> Mental / Environmental
Else no derived centers -> Lunar
Else -> Unknown / Needs Review with validation warning
```

Do not overclaim correctness for Ego, Self-Projected, or Mental Projector authority unless explicit deterministic rules and tests are implemented.

If an authority case is unsupported or ambiguous, return either:

```text
authority = "Needs Review"
```

or:

```text
authority = "Unknown"
```

and add a validation warning explaining why.

### Strategy, Not-Self Theme, Signature

Derive from type:

```text
Generator -> Responding, Frustration, Satisfaction
Manifesting Generator -> Responding, Frustration, Satisfaction
Manifestor -> Informing, Anger, Peace
Projector -> Waiting for Invitation, Bitterness, Success
Reflector -> Waiting a Lunar Cycle, Disappointment, Surprise
```

If type is unknown or needs review, these fields should also be conservative or blank with warnings.

## Validation Rules

Validation should be its own module and should happen before evaluation and CLI final output.

Validation must check:

- malformed or missing planetary activations
- missing Personality Sun or Design Sun
- missing required non-Sun Personality or Design planetary activations
- invalid gate numbers outside 1-64
- invalid line numbers outside 1-6
- disagreement between column-derived active gates and visually detected active gates
- visible colored channels not present in derived active channels
- visually defined centers versus `derived_chart_data.defined_centers`
- Reflector consistency:
  - no derived defined centers
  - no active channels
  - Lunar authority
  - Reflector type
- unsupported authority cases should produce warnings rather than hallucinated authority

Warnings should have machine-readable codes, severity, validity effect, and source so tests and evaluation can compare expected warning behavior.

Default warning contract:

| Code | Severity | affects_validity | Meaning |
|---|---|---:|---|
| `VISIBLE_CHANNEL_NORMALIZED` | `INFO` | false | A valid reversed visible channel was normalized to canonical orientation. |
| `INVALID_VISIBLE_CHANNEL` | `WARNING` | false | Vision reported an impossible or non-canonical visible channel. |
| `VISIBLE_CHANNEL_NOT_DERIVED` | `WARNING` | false | Vision reported a visible channel that was not derived from column activations. |
| `DERIVED_CHANNEL_NOT_VISIBLE` | `WARNING` | false | A column-derived channel was not detected visually. |
| `VISUALLY_ACTIVE_GATES_MISMATCH` | `WARNING` | false | Visual gate evidence differs from column-derived active gates. |
| `VISUALLY_DEFINED_CENTERS_MISMATCH` | `WARNING` | false | Visual center evidence differs from derived centers. |
| `UNSUPPORTED_AUTHORITY` | `WARNING` | false | The current deterministic authority rules do not safely support the case. |
| `MISSING_PERSONALITY_SUN` | `ERROR` | true | Personality Sun is absent or unusable, so profile cannot be fully derived. |
| `MISSING_DESIGN_SUN` | `ERROR` | true | Design Sun is absent or unusable, so profile cannot be fully derived. |
| `MISSING_ACTIVATION` | `ERROR` | true | A required non-Sun Personality or Design planetary activation is absent, blank, null, or unreadable. |
| `MALFORMED_ACTIVATION` | `ERROR` | true | A required planetary activation cannot be interpreted as Gate.Line. |
| `INVALID_ACTIVATION_GATE` | `ERROR` | true | An activation gate is outside 1 through 64. |
| `INVALID_ACTIVATION_LINE` | `ERROR` | true | An activation line is outside 1 through 6. |

Not every warning makes the full extraction invalid. Parser errors should be used for invalid JSON and missing required structural fields. Recoverable Vision extraction issues should generally survive parsing and become validation warnings so evaluation can measure them. Validation warnings should be used for cross-source disagreement between visual evidence and column-derived facts.

Rules:

- `validation.is_valid == not any(issue.affects_validity for issue in validation.warnings)`.
- Invalid JSON and missing required top-level structure remain hard parser errors.
- Recoverable extraction-quality issues remain structured issues that reach validation.
- An invalid visual channel alone must not invalidate an otherwise derivable chart.
- Visual disagreement alone must not invalidate an otherwise derivable chart.
- Missing or invalid mandatory planetary activation data required for profile or deterministic derivation must invalidate the result.
- Missing `personality.sun` produces `MISSING_PERSONALITY_SUN`, not a duplicate `MISSING_ACTIVATION`.
- Missing `design.sun` produces `MISSING_DESIGN_SUN`, not a duplicate `MISSING_ACTIVATION`.
- Missing any other required Personality or Design activation produces `MISSING_ACTIVATION`.

## Configuration

Add Phase 2 config separately from Phase 1 RAG config or clearly namespace it.

Prefer OpenAI-only config for Phase 2 unless provider abstraction is truly needed.

Environment variables:

```env
HD_VISION_MODEL=
HD_VISION_REAL_API=0
HD_BODYGRAPH_SAMPLE_DIR=data/bodygraph_samples/images
HD_BODYGRAPH_GOLDEN_LABELS=data/bodygraph_samples/golden_labels.example.json
OPENAI_API_KEY=
```

Optional only if provider abstraction is intentionally implemented:

```env
HD_VISION_PROVIDER=openai
```

Default tests must not require API keys.

Real Vision API calls must be opt-in only with:

```text
HD_VISION_REAL_API=1
```

## Data and Privacy Rules

- Do not commit `.env`.
- Do not commit API keys.
- Do not log API keys.
- Do not log full base64 images.
- Do not commit private user BodyGraph images by default.
- Do not commit generated raw Vision responses containing private data unless sanitized.
- Do not call real Vision APIs during default tests.
- Do not create cloud resources in Phase 2.
- Do not write large image/base64 dumps to git.

Add `.gitignore` entries if needed:

```gitignore
data/bodygraph_samples/images/
data/bodygraph_samples/private/
*.vision_response.json
*.bodygraph_prediction.json
```

Allow committing:

- `data/bodygraph_samples/golden_labels.example.json`
- small sanitized JSON fixtures under `tests/fixtures/bodygraph/`
- `tests/fixtures/bodygraph/test1.png` as the only committed image fixture for clean-repo mock/offline tests

`data/bodygraph_samples/images/` is for local/private real chart images and is ignored by git. `data/bodygraph_samples/private/` is also ignored by git. Do not place private user BodyGraph images under `tests/fixtures/`.

## Golden Labels

Golden labels should include both raw expected labels and derived expected labels. They should also include warning expectations and metadata.

Semantics:

- `expected_raw` evaluates Vision extraction of raw visible facts.
- `expected_derived` evaluates deterministic Python derivation.
- `expected_validation` evaluates warning behavior and validity status.

Example shape:

```json
{
  "cases": [
    {
      "id": "test1_generator_emotional_4_6",
      "image_file": "test1.png",
      "label_source": "manual",
      "notes": "Non-private local sample label",
      "expected_raw": {
        "personality": {
          "sun": "61.4",
          "earth": "62.4",
          "north_node": "3.1",
          "south_node": "60.2",
          "moon": "10.3",
          "mercury": "34.2",
          "venus": "30.1",
          "mars": "41.5",
          "jupiter": "50.2",
          "saturn": "59.2",
          "uranus": "18.1",
          "neptune": "22.2",
          "pluto": "14.3"
        },
        "design": {
          "sun": "32.6",
          "earth": "42.6",
          "north_node": "5.1",
          "south_node": "44.1",
          "moon": "60.3",
          "mercury": "3.4",
          "venus": "34.1",
          "mars": "10.4",
          "jupiter": "61.1",
          "saturn": "62.1",
          "uranus": "41.1",
          "neptune": "14.1",
          "pluto": "50.1"
        },
        "visually_defined_centers": ["G", "Sacral", "Solar Plexus", "Root"],
        "visually_undefined_centers": ["Head", "Ajna", "Throat", "Ego", "Spleen"],
        "visually_active_gates": [],
        "visible_colored_channels": ["10-34", "3-60", "30-41"]
      },
      "expected_derived": {
        "active_gates": [3, 5, 10, 14, 18, 22, 30, 32, 34, 41, 42, 44, 50, 59, 60, 61, 62],
        "active_channels": ["10-34", "3-60", "30-41"],
        "defined_centers": ["G", "Sacral", "Solar Plexus", "Root"],
        "type": "Generator",
        "authority": "Emotional / Solar Plexus",
        "profile": "4/6",
        "definition": "Single Definition",
        "strategy": "Responding",
        "not_self_theme": "Frustration",
        "signature": "Satisfaction"
      },
      "expected_validation": {
        "is_valid": true,
        "expected_warning_codes": []
      }
    }
  ]
}
```

Golden-label rules:

- Use exactly the 13 canonical planetary field names shown in the example for both `personality` and `design`.
- Do not duplicate the planetary columns with convenience fields such as `personality_sun` or `design_sun`.
- Every golden-label case used for derived active-gate, active-channel, center, profile, type, authority, or definition evaluation must contain manually verified values for all 26 planetary activations.
- A case with unavailable planetary source data may be included only as a raw visual-evidence case.
- A partial case must explicitly declare an evaluation scope that excludes it from derived-chart metrics.
- Do not silently treat missing golden activations as empty values, false values, or evaluation mismatches.
- Confidence and `uncertain_items` may be included in `expected_raw` when useful for parser and CLI fixtures.
- Phase 2 golden-label evaluation does not need to score confidence calibration yet.

The sample set should recommend coverage for:

- Generator
- Manifesting Generator
- Projector
- Manifestor
- Reflector
- Single Definition
- Split Definition
- No Definition
- ambiguous / invalid cases

Add extra Reflector cases because Vision models often misread Reflectors.

Unit tests can use synthetic/mock JSON. Vision prompt evaluation should use real-ish local chart images ignored by git. Non-private sample images are optional.

## Evaluation Metric Semantics

Set-valued fields should use precision, recall, and F1:

- visually defined centers
- derived defined centers
- active gates
- visually active gates
- active channels
- visible colored channels
- warning codes

Calculate precision, recall, and F1 per chart. Report macro-average across charts by default. Micro-average may be added if it is clearly labeled.

Raw planetary activation extraction metrics:

- `personality_activation_exact_match_rate`
- `design_activation_exact_match_rate`
- `activation_exact_match_rate`

For each eligible chart, each rate is:

```text
number of exact activation matches / number of expected activation fields in scope
```

Report the macro-average across eligible charts by default. Activation exact match includes both gate number and line number. Gate-only F1 is not enough because an incorrect line can produce the same active gate while still being an incorrect Vision extraction.

Exact-match fields:

- profile
- type
- authority
- definition
- strategy
- not_self_theme
- signature

`overall_basic_info_accuracy` is the average of exact-match booleans across all listed basic-info fields. Warning-code behavior should be reported as warning-code precision, recall, and F1 rather than a vague match rate.

## Testing Strategy

- Default tests must not call OpenAI.
- Default tests must not require API keys.
- Default tests must not require private images.
- Default tests must not require Phase 1 Chroma storage.
- Tests should use mocked Vision responses.
- Tests should cover parser behavior.
- Tests should cover deterministic interpreter behavior.
- Tests should cover validation behavior.
- Tests should cover evaluation metrics.
- Use TDD where practical.

Real Vision API tests must be manual / opt-in only with `HD_VISION_REAL_API=1`.

## Recommended Implementation Skills

During planning, use:

- `human-design-rag-phase-discipline`
- `writing-plans`

During implementation, use:

- `human-design-rag-module-conventions`
- `human-design-rag-privacy-and-cost-rules`

During tests, API key handling, Vision API calls, image privacy, and `.gitignore` work, use:

- `human-design-rag-privacy-and-cost-rules`
- `verification-before-completion`

## Implementation Tasks

Create Phase 2 tasks starting at Task 16 because Phase 1 ended at Task 15.

Task dependency chain:

```text
Task 16 -> Task 16A -> Task 18 parser -> Task 19 interpreter -> Task 20 validation
```

Task 17 fixture and golden-label work may begin after Task 16, but it must follow the final raw-versus-derived schema semantics. Task 22 depends on finalized golden labels, parser behavior, interpreter behavior, and validation behavior.

### Task 16: Define BodyGraph extraction schema

Goal:
Define typed data models for raw Vision extraction, derived chart data, confidence/uncertainty, and validation results.

Files touched:
- `src/human_design/vision/__init__.py`
- `src/human_design/vision/models.py`
- `tests/test_vision_models.py`

Dependencies:
- `pytest`
- `pydantic` only if the implementation chooses it for schema validation

Acceptance criteria:
- Models exist for:
  - planetary activation value
  - Personality activation column
  - Design activation column
  - raw Vision extraction result
  - confidence / uncertainty result
  - derived basic info
  - derived chart data
  - validation warning
  - validation result
  - full BodyGraph extraction result
- Raw Vision result includes:
  - `personality`
  - `design`
  - `visually_defined_centers`
  - `visually_undefined_centers`
  - `visually_active_gates`
  - `visible_colored_channels`
  - `confidence`
  - `uncertain_items`
- Activation columns contain all 13 canonical planetary keys for Personality and all 13 canonical planetary keys for Design.
- Activation values can represent `Gate.Line` values or unavailable values with `None` or an equivalent typed nullable representation.
- Derived basic info includes:
  - `type`
  - `authority`
  - `profile`
  - `strategy`
  - `definition`
  - `not_self_theme`
  - `signature`
- Derived chart data includes:
  - `basic_info`
  - `active_gates`
  - `active_channels`
  - `defined_centers`
- Validation includes:
  - `is_valid`
  - `warnings`
- Each validation warning or issue includes:
  - `code`
  - `message`
  - `severity`
  - `affects_validity`
  - `source`
- Validation severity supports exactly `INFO`, `WARNING`, and `ERROR`.
- Validation source supports exactly `parser`, `interpreter`, and `validation`.
- `validation.is_valid` is derived from `not any(issue.affects_validity for issue in warnings)`.
- A typed parser result can preserve parser normalization warnings and merge them into final validation output, for example `ParseResult(raw_vision=RawVisionExtraction, warnings=tuple[ValidationWarning, ...])`.
- Validation warnings have machine-readable codes.
- Validation warnings include the generic `MISSING_ACTIVATION` code for unavailable non-Sun required activation values.
- Missing Sun fields use `MISSING_PERSONALITY_SUN` or `MISSING_DESIGN_SUN` without duplicate `MISSING_ACTIVATION` warnings.
- Confidence values are finite floats in the inclusive range `0.0` to `1.0`.
- Confidence values are not percentages, strings, or free-form text.
- `uncertain_items` include `field_path`, `observed_value`, `reason`, and numeric `confidence`.
- `field_path` in `uncertain_items` points only to raw Vision fields.
- Models use type hints.
- Models avoid importing Vision API clients.
- Tests do not call any Vision API.

Out of scope:
- Do not implement prompt loading.
- Do not implement parser behavior.
- Do not implement interpreter rules.
- Do not call OpenAI.
- Do not add CLI behavior.
- Do not create sample image files.

Testing requirements:
- Tests should instantiate the models with representative raw and derived chart data.
- Tests should verify optional uncertainty/confidence fields can represent ambiguous raw extraction.
- Tests should verify confidence values reject or clearly flag values outside `0.0` to `1.0`.
- Tests should verify validation warning codes, severity, source, and validity effects are available for assertions.
- Tests should verify `is_valid` is false when any issue has `affects_validity=True`.
- Tests should verify unavailable non-Sun activation values can be represented and produce `MISSING_ACTIVATION`.
- Tests should verify missing Sun fields do not also produce duplicate `MISSING_ACTIVATION` warnings.
- Tests should verify `is_valid` is false when a required non-Sun activation is unavailable.
- Tests should not require real images, real API keys, or Phase 1 Chroma storage.

Verification:

```sh
uv run pytest tests/test_vision_models.py
uv run ruff check .
```

### Task 16A: Define canonical BodyGraph domain constants

Goal:
Create stable domain constants needed by parsing, interpretation, validation, and evaluation before parser implementation begins.

Files touched:
- `src/human_design/vision/constants.py`
- `tests/test_bodygraph_constants.py`

Dependencies:
- Task 16 models
- `pytest`

Acceptance criteria:
- Define exactly the nine canonical centers:
  - `Head`
  - `Ajna`
  - `Throat`
  - `G`
  - `Ego`
  - `Sacral`
  - `Spleen`
  - `Solar Plexus`
  - `Root`
- Define parser aliases:
  - `Heart`, `Will`, `Ego` normalize to `Ego`.
  - `G Center`, `Self`, `Identity` normalize to `G`.
  - `Emotional`, `Solar Plexus` normalize to `Solar Plexus`.
- Define exactly the canonical 36 Human Design channels already listed in this plan.
- Define `CHANNEL_TO_CENTERS` exactly as specified in the Canonical Channel-to-Center Mapping table.
- `CHANNEL_TO_CENTERS` keys exactly equal `ALL_CHANNELS`.
- Every channel maps to exactly two distinct canonical center names.
- No alternate or duplicate channel-to-center mapping exists elsewhere in Phase 2 code.
- Define `MOTOR_CENTERS`.
- Tests verify:
  - exactly 36 canonical channels exist
  - the canonical channel set exactly matches the 36 channels listed in this plan
  - the channel-to-center mapping exactly covers the table in this plan
  - every canonical channel maps to exactly two canonical centers
  - all mapped center names are canonical
  - aliases normalize deterministically
- No parser behavior, Vision API code, CLI behavior, image handling, or interpretation logic is added in this task.

Out of scope:
- Do not parse raw JSON.
- Do not derive active gates, channels, centers, or `basic_info`.
- Do not call OpenAI or any Vision API.
- Do not add CLI behavior or image handling.

Testing requirements:
- Tests should assert exact center vocabulary and channel count.
- Tests should assert exact channel-to-center table coverage, including at least `3-60`, `10-34`, `10-57`, `20-34`, `21-45`, `24-61`, and `37-40`.
- Tests should assert every channel endpoint mapping uses canonical center names.
- Tests should assert aliases normalize deterministically.

Verification:

```sh
uv run pytest tests/test_bodygraph_constants.py
uv run ruff check .
```

### Task 17: Prepare sample images and golden labels

Goal:
Define the local sample image and golden label structure for BodyGraph extraction evaluation.

Files touched:
- `data/bodygraph_samples/golden_labels.example.json`
- `tests/fixtures/bodygraph/`
- `.gitignore`
- `tests/test_bodygraph_evaluation.py` if needed

Dependencies:
- `pytest`

Acceptance criteria:
- A documented golden label JSON format exists.
- Golden labels include:
  - case id
  - image filename
  - label source
  - notes
  - expected raw labels
  - expected derived labels
  - expected validation result
  - expected warning codes
- Raw labels include:
  - full Personality activation column with the 13 canonical planetary field names
  - full Design activation column with the 13 canonical planetary field names
  - visually active gates if available
  - visually defined centers
  - visually undefined centers
  - visible full colored channels
  - confidence values when included
  - uncertain items when included
- Derived labels include:
  - active gates derived from columns
  - active channels
  - defined centers derived from active channel endpoints
  - type
  - authority
  - profile
  - definition
  - strategy
  - not_self_theme
  - signature
- Every case used for derived active-gate, active-channel, center, profile, type, authority, or definition evaluation includes manually verified values for all 26 planetary activations.
- Partial cases with unavailable planetary source data are allowed only as raw visual-evidence cases.
- Partial cases explicitly declare an evaluation scope that excludes them from derived-chart metrics.
- Missing golden activations are not silently treated as empty values, false values, or evaluation mismatches.
- The sample plan recommends covering Generator, Manifesting Generator, Projector, Manifestor, and Reflector.
- The sample plan recommends extra Reflector examples.
- Private images are ignored by git by default.
- Synthetic/mock JSON fixtures are used for unit tests.
- `tests/fixtures/bodygraph/test1.png` exists as a small non-private or synthetic fixture image.
- Mock/offline CLI tests use `tests/fixtures/bodygraph/test1.png`, not ignored private image paths.
- Private evaluation images remain under `data/bodygraph_samples/images/` or `data/bodygraph_samples/private/` and are ignored by git.
- The fixture image must be safe to commit.
- Real-ish chart images are allowed for local evaluation but should be ignored by git unless non-private.
- `.gitignore` protects private images and generated Vision responses if those patterns are not already present.

Out of scope:
- Do not call any Vision API.
- Do not commit private BodyGraph images.
- Do not create a manual annotation UI.
- Do not implement evaluation logic unless a small fixture-reading test requires it.

Testing requirements:
- Tests may validate that example golden labels load as JSON and contain required keys.
- Tests should verify full derived-evaluation cases contain all 26 planetary activations.
- Tests should verify partial raw-only cases declare an evaluation scope excluding derived-chart metrics.
- Tests should not depend on private local images existing.
- Tests should not call OpenAI or require API keys.

Verification:

```sh
uv run pytest
uv run ruff check .
git status --short
```

### Task 18: Build raw Vision extraction prompt and parser

Goal:
Create the prompt and parser that ask the Vision model to extract only raw visible facts, not final Human Design interpretations.

Files touched:
- `prompts/bodygraph_raw_extraction.txt`
- `src/human_design/vision/prompt.py`
- `src/human_design/vision/parser.py`
- `tests/test_bodygraph_parser.py`

Dependencies:
- Task 16 models
- Task 16A constants
- `pytest`

Acceptance criteria:
- Prompt explicitly tells the model:
  - extract only raw visible facts
  - do not infer type
  - do not infer authority
  - do not infer profile
  - do not infer definition
  - do not infer strategy
  - do not infer not_self_theme
  - do not infer signature
- Prompt asks for:
  - Personality column values
  - Design column values
  - visually colored / defined centers
  - visually undefined centers if visible/inferable
  - visually active gates
  - visibly full colored channels
  - numeric confidence values per activation and major visual extraction group
  - uncertainty items for ambiguous raw visual observations
- Prompt requests every canonical Personality and Design activation key.
- Prompt instructs the model to use JSON `null` when a required activation cannot be read.
- Prompt forbids confidence percentages, strings such as `"high"` or `"low"`, and free-form confidence text.
- Prompt requires strict JSON only.
- Parser can parse valid model JSON into typed models.
- Parser rejects malformed JSON with clear errors.
- Parser raises clear parse errors for missing required top-level schema fields.
- Parser raises clear parse errors for missing `personality` or `design` containers.
- Parser normalizes planetary values like `61.4`.
- Parser preserves JSON `null` activation values as unavailable values.
- Parser treats blank activation strings as unavailable values.
- Parser normalizes omitted individual activation keys inside otherwise valid Personality or Design data and preserves them as structured missing issues.
- Parser does not treat omitted individual activation keys as unrecoverable parser failures.
- Parser preserves missing non-Sun activations so validation can emit `MISSING_ACTIVATION`.
- Parser preserves missing `personality.sun` and `design.sun` so validation can emit their Sun-specific codes without duplicate `MISSING_ACTIVATION`.
- Parser preserves malformed non-empty activation values so validation can emit `MALFORMED_ACTIVATION`.
- Parser normalizes channel strings such as `3-60`.
- Parser normalizes gate numbers as integers.
- Parser validates every `visible_colored_channel` against `ALL_CHANNELS`.
- Parser normalizes reversed valid visible channel strings to canonical orientation.
- Parser preserves parser warning information for reversed valid channels such as `VISIBLE_CHANNEL_NORMALIZED`.
- Parser preserves impossible visible channel issues so validation can emit `INVALID_VISIBLE_CHANNEL`.
- Parser parses planetary activation values into gate and line.
- Parser validates activation gate range 1-64.
- Parser validates activation line range 1-6.
- Parser preserves invalid activation issues so validation can emit `INVALID_ACTIVATION_GATE` or `INVALID_ACTIVATION_LINE`.
- Parser validates confidence values as finite floats in the inclusive range `0.0` to `1.0`.
- Parser rejects or clearly flags confidence values outside the allowed range.
- Parser preserves `uncertain_items` with `field_path`, `observed_value`, `reason`, and numeric `confidence`.
- Parser rejects or clearly flags `uncertain_items.field_path` values that point to derived chart fields.
- Parser does not derive `basic_info`.
- Parser does not derive active channels.
- Default tests use mocked JSON responses and no Vision API calls.

Out of scope:
- Do not implement a Vision API client.
- Do not derive type, authority, profile, definition, strategy, not_self_theme, or signature.
- Do not inspect image files.
- Do not call OpenAI.
- Do not write raw model responses to disk.

Testing requirements:
- Tests should cover valid strict JSON.
- Tests should cover malformed JSON and missing required top-level fields.
- Tests should cover missing `personality` and `design` containers as hard parser errors.
- Tests should cover normalization of channels, gates, and planetary activation values.
- Tests should cover JSON `null` activation values as preserved unavailable values.
- Tests should cover blank activation strings as unavailable values.
- Tests should cover omitted individual activation keys normalized and preserved as structured missing issues.
- Tests should cover malformed non-empty values such as `"unknown"` and `"61.x"` as `MALFORMED_ACTIVATION`.
- Tests should cover `60.1` -> gate `60`, line `1`.
- Tests should cover `50.4` -> gate `50`, line `4`.
- Tests should cover `60.9` as invalid line.
- Tests should cover `99.1` as invalid gate.
- Tests should cover `57-10` normalizing to `10-57`.
- Tests should cover `3 - 60` normalizing to `3-60`.
- Tests should cover `34-99` and `10-99` as invalid channels.
- Tests should cover `10-34` as valid.
- Tests should cover valid numeric confidence values.
- Tests should cover invalid confidence values below `0.0`, above `1.0`, non-finite floats, and string confidence values.
- Tests should cover `uncertain_items` with raw Vision field paths.
- Tests should verify parser warnings can be merged into final validation output.
- Tests should confirm final Human Design concepts are not accepted as Vision-derived facts.
- Tests should not require private images or API keys.

Verification:

```sh
uv run pytest tests/test_bodygraph_parser.py
uv run ruff check .
```

### Task 19: Build deterministic BodyGraph interpreter for `basic_info`

Goal:
Derive active gates, active channels, and `basic_info` from raw Vision facts using deterministic Python rules.

Files touched:
- `src/human_design/vision/interpreter.py`
- `tests/test_bodygraph_interpreter.py`

Dependencies:
- Task 16 models
- Task 16A constants
- Task 18 parser-normalized raw facts
- `pytest`

Acceptance criteria:
- Uses the `ALL_CHANNELS`, channel-to-center mapping, canonical centers, and `MOTOR_CENTERS` from Task 16A.
- Active gates are derived from all Personality and Design planetary activation gate numbers.
- Active channels are derived from active gates using `ALL_CHANNELS`.
- Interpreter derives `active_channels` only by iterating over `ALL_CHANNELS`.
- A derived channel is active only when both gates are present in the column-derived `active_gates`.
- Derived `active_channels` never include channels outside `ALL_CHANNELS`.
- Vision-reported `visible_colored_channels` do not directly become derived `active_channels`.
- Vision-reported `visible_colored_channels` are used only as supporting evidence for validation.
- `derived_chart_data.defined_centers` are derived from active channel endpoint centers.
- `derived_chart_data.defined_centers` equals the canonical-order union of all endpoint centers from `active_channels`.
- Type, authority, definition, strategy, not_self_theme, and signature use derived chart data, not raw visual center observations.
- Profile is derived from Personality Sun line and Design Sun line.
- Definition is derived using the Phase 2 v1 coarse connected-component taxonomy over `derived_chart_data.defined_centers` and active channels.
- Exactly two disconnected derived components produce `Split Definition`; Phase 2 v1 does not distinguish `Simple-Split Definition` from `Wide-Split Definition` and does not implement bridge-gate analysis.
- Type mapping logic requires a directly active canonical motor-to-Throat channel for Manifesting Generator or Manifestor classification.
- Multi-hop graph connectivity through intermediate centers does not by itself qualify as a motor-to-Throat connection for Type classification.
- A Sacral-defined chart with a direct active motor-to-Throat channel is a Manifesting Generator; without one, it is a Generator.
- A Sacral-undefined chart with a direct active motor-to-Throat channel is a Manifestor, subject to the remaining Type rules.
- Authority priority logic exists.
- Strategy, not_self_theme, and signature are derived from type.
- Unsupported authority edge cases return `Unknown` or `Needs Review` with warnings rather than hallucinating.
- Interpreter does not call Vision API.
- Interpreter does not use an LLM.
- Tests cover:
  - Generator
  - Manifesting Generator
  - Projector
  - Manifestor
  - Reflector
  - Single Definition
  - Split Definition
  - No Definition
  - exactly two disconnected derived components produce the coarse Phase 2 v1 `Split Definition` output without Simple-Split versus Wide-Split classification
  - a direct active motor-to-Throat channel qualifies for Manifesting Generator or Manifestor classification under the remaining Type rules
  - a multi-hop active-channel path ending at `Throat`, such as `2-14` plus `1-8`, does not qualify by itself as a motor-to-Throat connection
  - profile derivation such as `61.4` + `32.6` -> `4/6`
  - gate derivation from planetary columns
  - channel derivation such as gates `3` and `60` -> `3-60`
  - active gates `{3, 60}` derive `3-60`
  - active gates `{10, 57}` derive `10-57`
  - active gates `{34, 57}` derive `34-57`
  - impossible fake channels are never derived
  - derived defined centers come from active channel endpoints
  - `active_channels = []` derives `defined_centers = []`
  - `active_channels = ["3-60"]` derives `defined_centers = ["Sacral", "Root"]`
  - `active_channels = ["3-60", "10-34", "30-41"]` derives `defined_centers = ["G", "Sacral", "Solar Plexus", "Root"]`
  - channel mapping `3-60` -> `Sacral`, `Root`
  - channel mapping `10-34` -> `G`, `Sacral`
  - channel mapping `10-57` -> `G`, `Spleen`
  - channel mapping `20-34` -> `Throat`, `Sacral`
  - channel mapping `21-45` -> `Ego`, `Throat`
  - channel mapping `24-61` -> `Ajna`, `Head`
  - channel mapping `37-40` -> `Solar Plexus`, `Ego`
  - conservative authority fallback

Out of scope:
- Do not call Vision APIs.
- Do not parse raw JSON directly.
- Do not validate visual disagreements yet.
- Do not generate readings.
- Do not modify Phase 1 RAG, Chroma, or retrieval modules.

Testing requirements:
- Tests should build in-memory raw model objects or parsed fixture objects.
- Constants tests from Task 16A should verify `ALL_CHANNELS` has exactly 36 channels.
- Tests should verify derived active channels can only come from `ALL_CHANNELS`.
- Tests should verify derived defined centers equal the canonical-order union of active channel endpoint centers.
- Tests should cover graph connected components for the coarse Phase 2 v1 Definition taxonomy, including exactly two disconnected components producing `Split Definition`.
- Tests should not add `Wide-Split Definition` output or bridge-gate logic.
- Tests should cover direct active canonical motor-to-Throat channel logic for Type classification.
- Tests should distinguish a qualifying direct active motor-to-Throat channel from a non-qualifying multi-hop graph path ending at `Throat`.
- Tests should cover unsupported authority warnings.
- Tests should not use an LLM or require API keys.

Verification:

```sh
uv run pytest tests/test_bodygraph_constants.py tests/test_bodygraph_interpreter.py
uv run ruff check .
```

### Task 20: Build validation module and offline validation tests

Goal:
Validate consistency between raw Vision facts, deterministic derived chart data, and visual evidence.

Files touched:
- `src/human_design/vision/validation.py`
- `tests/test_bodygraph_validation.py`
- possible fixture JSON files under `tests/fixtures/bodygraph/`

Dependencies:
- Task 16 models
- Task 16A constants
- Task 18 parser
- Task 19 interpreter
- `pytest`

Acceptance criteria:
- Validation can run after parser and interpreter.
- Validation produces structured warnings with machine-readable codes.
- Validation checks:
  - missing Personality Sun
  - missing Design Sun
  - missing required non-Sun Personality or Design activation
  - malformed activation values
  - invalid gate numbers
  - invalid line numbers
  - Vision-reported `visible_colored_channels` against canonical `ALL_CHANNELS`
  - Vision-reported `visible_colored_channels` against interpreter-derived `active_channels`
  - disagreement between column-derived active gates and visually active gates
  - visible colored channels not present in derived active channels
  - visually defined centers versus `derived_chart_data.defined_centers`
  - Reflector consistency
  - unknown / needs-review authority cases
- Validation emits machine-readable warning codes for invalid, normalized, missing, or extra channels.
- Validation emits severity and `affects_validity` according to the default warning contract.
- Validation computes `is_valid` from `not any(issue.affects_validity for issue in warnings)`.
- Missing non-Sun activation emits `MISSING_ACTIVATION`.
- Missing Personality Sun emits only `MISSING_PERSONALITY_SUN`.
- Missing Design Sun emits only `MISSING_DESIGN_SUN`.
- Missing or invalid required activation data makes `is_valid` false.
- Validation compares Vision-reported `visible_colored_channels` against canonical `ALL_CHANNELS`.
- Validation compares Vision-reported `visible_colored_channels` against interpreter-derived `active_channels`.
- Validation merges parser warning information such as `VISIBLE_CHANNEL_NORMALIZED`, `INVALID_VISIBLE_CHANNEL`, `MISSING_ACTIVATION`, `MALFORMED_ACTIVATION`, `INVALID_ACTIVATION_GATE`, and `INVALID_ACTIVATION_LINE` into final validation warnings.
- Invalid visual channels and visual disagreement alone do not invalidate an otherwise derivable chart.
- Missing or invalid mandatory planetary activation data required for profile or deterministic derivation invalidates the result.
- Validation does not call Vision API.
- Validation does not use an LLM.
- Tests include:
  - valid Generator example
  - valid Reflector-like example
  - invalid visible channel
  - reversed valid visible channel
  - visible channel not derived
  - derived channel not visible
  - mismatch between visually colored channel and derived active channels
  - missing Sun activation
  - missing non-Sun activation
  - unsupported authority warning

Out of scope:
- Do not call real Vision APIs.
- Do not implement evaluation metrics.
- Do not implement CLI behavior.
- Do not generate readings.

Testing requirements:
- Tests should assert warning codes rather than fragile warning text where possible.
- Tests should assert warning severity, source, and `affects_validity`.
- Tests should check `is_valid` for both valid and invalid cases.
- Tests should verify an invalid visual channel does not make `is_valid` false by itself.
- Tests should verify missing Personality Sun, missing Design Sun, malformed activation, invalid activation gate, and invalid activation line make `is_valid` false.
- Tests should verify missing non-Sun activation emits `MISSING_ACTIVATION` and makes `is_valid` false.
- Tests should verify missing Personality Sun emits only `MISSING_PERSONALITY_SUN`.
- Tests should verify missing Design Sun emits only `MISSING_DESIGN_SUN`.
- Tests should verify visual-vs-derived center mismatch emits `VISUALLY_DEFINED_CENTERS_MISMATCH` without invalidating the chart by itself.
- Tests should verify unsupported authority cases are warning-producing, not silently accepted.
- Tests should not require private images, API keys, or Phase 1 Chroma.

Verification:

```sh
uv run pytest tests/test_bodygraph_validation.py tests/test_bodygraph_parser.py tests/test_bodygraph_interpreter.py
uv run ruff check .
```

### Task 21: Add mocked end-to-end pipeline tests

Goal:
Add robust offline tests for parser -> interpreter -> validation using mocked Vision JSON.

Files touched:
- `tests/test_bodygraph_parser.py`
- `tests/test_bodygraph_interpreter.py`
- `tests/test_bodygraph_validation.py`
- possible fixture JSON files under `tests/fixtures/bodygraph/`

Dependencies:
- Task 16 models
- Task 16A constants
- Task 18 parser
- Task 19 interpreter
- Task 20 validation
- `pytest`

Acceptance criteria:
- Mocked raw Vision JSON can be parsed.
- Parsed raw Vision facts can be passed into the interpreter.
- Interpreter returns expected `basic_info`.
- Validation returns expected warnings.
- Tests include at least:
  - Personality Sun `61.4`, Design Sun `32.6`, expected profile `4/6`
  - active gates including `3` and `60`, expected active channel `3-60`
  - active gates including `10` and `34`, expected active channel `10-34`
  - Reflector-like chart with no derived defined centers, expected Reflector and Lunar authority
- Tests do not call real Vision APIs.
- Tests do not require private images.

Out of scope:
- Do not add real Vision API tests.
- Do not add CLI code.
- Do not add evaluation scripts.
- Do not modify Phase 1 RAG logic.

Testing requirements:
- Fixture JSON should be small and sanitized.
- Tests should assert both derived chart fields and validation warning behavior.
- Tests should avoid real image files unless they are small, non-private, and intentionally committed.

Verification:

```sh
uv run pytest tests/test_bodygraph_parser.py tests/test_bodygraph_interpreter.py tests/test_bodygraph_validation.py
uv run ruff check .
```

### Task 22: Add evaluation script for extraction accuracy

Goal:
Evaluate Vision extraction and deterministic interpretation against golden labels.

Files touched:
- `src/human_design/vision/evaluation.py`
- `scripts/evaluate_bodygraph_extraction.py`
- `tests/test_bodygraph_evaluation.py`

Dependencies:
- Finalized golden labels from Task 17
- Parser behavior from Task 18
- Interpreter behavior from Task 19
- Validation behavior from Task 20
- Offline pipeline fixtures from Task 21
- `pytest`

Acceptance criteria:
- Evaluation can compare predicted output to golden labels.
- Metrics include:
  - `personality_activation_exact_match_rate`
  - `design_activation_exact_match_rate`
  - `activation_exact_match_rate`
  - derived center precision, recall, and F1
  - visually defined center precision, recall, and F1 if available
  - active gate precision, recall, and F1
  - `visually_active_gate_f1` if available
  - active channel precision, recall, and F1
  - `visible_channel_f1` if available
  - `profile_exact_match`
  - `type_exact_match`
  - `authority_exact_match`
  - `definition_exact_match`
  - `strategy_exact_match`
  - `not_self_theme_exact_match`
  - `signature_exact_match`
  - `overall_basic_info_accuracy`
  - warning-code precision, recall, and F1
- Activation exact-match metrics compare both gate number and line number.
- Activation exact-match rates are calculated per chart as exact activation matches divided by expected activation fields in scope.
- Activation exact-match rates are macro-averaged across eligible charts by default.
- Full derived-evaluation cases require all 26 manually verified golden activations.
- Predictions with missing required activation values are evaluated as extraction failures, not silently excluded.
- Partial raw-only golden cases are excluded from derived-chart metrics only because their source labels are unavailable, not because a prediction is missing values.
- Set-valued metrics are calculated per chart and macro-averaged across charts by default.
- Micro-averages may be reported only if clearly labeled.
- `overall_basic_info_accuracy` is the average of exact-match booleans across profile, type, authority, definition, strategy, not_self_theme, and signature.
- Warning evaluation compares warning codes and, where useful, severity or `affects_validity` status.
- Evaluation script can run on mocked predictions or saved prediction JSON by default.
- Real Vision API evaluation is opt-in only.
- Output clearly shows per-case and aggregate metrics.
- If metrics fall below thresholds, script exits non-zero.
- Thresholds are configurable or constants.
- No real Vision API call in default tests.

Out of scope:
- Do not call real Vision APIs by default.
- Do not implement the extraction CLI in this task.
- Do not write private raw responses to git.
- Do not generate readings.

Testing requirements:
- Tests should use small golden-label and prediction fixtures.
- Tests should cover full 26-field planetary activation exact-match metrics.
- Tests should verify line mismatches fail activation exact match even when the gate matches.
- Tests should verify missing prediction activations count as extraction failures for eligible full-label cases.
- Tests should verify partial raw-only cases are excluded from derived-chart metrics.
- Tests should cover exact-match metrics and F1 metrics.
- Tests should cover warning-code precision, recall, F1, and severity or `affects_validity` comparison.
- Tests should cover threshold failure exit behavior through a function or script-level test.
- Tests should not require private images or API keys.

Verification:

```sh
uv run pytest tests/test_bodygraph_evaluation.py
uv run ruff check .
```

### Task 23: Add local CLI smoke test for one chart image

Goal:
Create a local CLI that extracts one BodyGraph image into raw Vision facts, derived chart data, and validation output.

Files touched:
- `scripts/extract_bodygraph.py`
- `src/human_design/vision/config.py`
- `src/human_design/vision/client.py`
- `tests/test_extract_bodygraph.py`
- `tests/test_vision_config.py`
- `.env.example`, only if Phase 2 configuration variables are not already documented there

Dependencies:
- Tasks 16 through 22
- OpenAI SDK only if real Vision client support is implemented
- `python-dotenv` if reused for config loading
- `pillow` only if basic image validation is needed
- `pytest`

Acceptance criteria:
- CLI accepts an image path.
- CLI supports `--mock-response` or similar offline mode.
- CLI supports real Vision API only with explicit opt-in, such as `HD_VISION_REAL_API=1`.
- CLI prints:
  - raw Vision extraction
  - derived chart data
  - validation warnings
- CLI does not print base64 image data.
- CLI does not print API keys.
- CLI exits clearly if real API mode is requested without API key.
- `.env.example` documents non-secret Phase 2 configuration defaults when they are absent.
- `.env.example` includes only non-secret placeholder or default values.
- `.env.example` must not contain a real API key, private image path, personal image filename, or private local directory.
- At minimum, Phase 2 documentation in `.env.example` should cover:
  - `HD_VISION_MODEL=`
  - `HD_VISION_REAL_API=0`
  - `HD_BODYGRAPH_SAMPLE_DIR=data/bodygraph_samples/images`
  - `HD_BODYGRAPH_GOLDEN_LABELS=data/bodygraph_samples/golden_labels.example.json`
- `OPENAI_API_KEY=` may remain as an empty placeholder only.
- `.env.example` changes belong to Task 23 because Task 23 introduces Phase 2 config and CLI behavior.
- Default tests mock client calls.
- Default tests do not call real Vision APIs.
- Default tests do not require private images.

Example commands:

```sh
uv run python scripts/extract_bodygraph.py tests/fixtures/bodygraph/test1.png --mock-response tests/fixtures/bodygraph/test1_raw_response.json

HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py data/bodygraph_samples/images/test1.png
```

The mock/offline command should work in a clean repo because it uses the committed non-private fixture image. The real API command is manual-only and requires the user to place a local non-private image in `data/bodygraph_samples/images/`.

Out of scope:
- Do not generate Human Design readings.
- Do not call Phase 1 retrieval.
- Do not write base64 image data to disk.
- Do not create a web app or API.
- Do not run real Vision API calls in default tests.

Testing requirements:
- Tests should monkeypatch or fake the Vision client.
- Tests should verify disabled real API behavior when `HD_VISION_REAL_API` is not set.
- Tests should verify mock-response mode can run without an API key.
- CLI tests should use `tests/fixtures/bodygraph/test1.png` for mock/offline mode.
- CLI tests must not depend on `data/bodygraph_samples/images/test1.png`.
- CLI tests must not require private local images.
- Tests should verify output does not contain API keys or base64 image data.
- Tests should not require private images.

Verification:

```sh
uv run pytest tests/test_extract_bodygraph.py tests/test_vision_config.py
uv run ruff check .
```

### Task 24: README update for Phase 2

Goal:
Document Phase 2 local Vision extraction scope, commands, privacy rules, and testing rules.

Files touched:
- `README.md`

Dependencies:
- Tasks 16 through 23

Acceptance criteria:
- README explains Phase 2 is local Vision extraction only.
- README explains raw Vision extraction vs deterministic interpreter.
- README explains active gates are derived from Personality/Design columns, with visual gates/channels used as supporting evidence.
- README documents environment variables.
- README documents default tests are offline and free.
- README documents real Vision API calls are opt-in only.
- README documents sample image / golden label layout.
- README documents CLI smoke test.
- README documents evaluation script.
- README warns not to commit private BodyGraph images, API keys, `.env`, raw base64 images, or private model responses.

Out of scope:
- Do not implement new Python code.
- Do not call real Vision APIs.
- Do not commit private images.
- Do not change Phase 1 retrieval behavior.

Testing requirements:
- Run the full default test suite after README updates.
- Confirm docs do not include secret values.

Verification:

```sh
uv run pytest
uv run ruff check .
git status --short
```

### Task 25: Final Phase 2 verification gate

Goal:
Verify Phase 2 locally with tests, linting, offline local pipeline execution, optional real Vision API testing, optional offline evaluation, and privacy checks.

Files touched:
- No new source files expected.
- Fix only issues found by verification.
- Documentation-only wording fixes are allowed if verification commands or README instructions are unclear.

Dependencies:
- Tasks 16 through 24
- Optional local BodyGraph image for manual real Vision testing
- Optional local golden labels and saved predictions for offline evaluation
- Real Vision API key only when intentionally running `HD_VISION_REAL_API=1`

Acceptance criteria:
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- Offline local pipeline test works with a saved mock Vision response.
- Offline local pipeline JSON output works.
- Real Vision API test remains documented as manual and opt-in only.
- Offline evaluation is documented as an optional developer accuracy check that requires matching golden labels and prediction `case_id`.
- No API keys, `.env`, private images, base64 dumps, private model responses, or private prediction files are committed.
- Phase 1 retrieval tests still pass.
- Phase 2 does not modify Phase 1 architecture.

Final Verification Gate:

```sh
uv run pytest
uv run ruff check .
```

Offline Local Pipeline Test:

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json
```

This command does not call OpenAI and does not require `OPENAI_API_KEY`. It loads a sanitized mock Vision response and verifies that the local parser, deterministic interpreter, validation logic, and CLI output formatting can run successfully.

Use `--json` to verify machine-readable output:

```sh
uv run python scripts/extract_bodygraph.py \
  tests/fixtures/bodygraph/test1.png \
  --mock-response tests/fixtures/bodygraph/test1_raw_response.json \
  --json
```

Manual Real Vision Test, optional only:

```sh
HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py \
  data/bodygraph_samples/images/test.png
```

This command is manual and opt-in. It requires a local BodyGraph image, a valid `OPENAI_API_KEY`, and intentional `HD_VISION_REAL_API=1`. It may create API cost.

Save real Vision output to private JSON:

```sh
mkdir -p data/bodygraph_samples/private

HD_VISION_REAL_API=1 uv run python scripts/extract_bodygraph.py \
  data/bodygraph_samples/images/test.png \
  --json > data/bodygraph_samples/private/test.bodygraph_prediction.json
```

Offline Evaluation, optional developer accuracy check:

Evaluation is not required for normal one-chart extraction. It is only meaningful when the saved prediction and golden-label file describe the same chart and use the same `case_id`.

```sh
uv run python scripts/evaluate_bodygraph_extraction.py \
  --golden-labels data/bodygraph_samples/private/golden_labels.local.json \
  --predictions data/bodygraph_samples/private/predictions.local.json
```

Do not compare a private real-chart prediction against `data/bodygraph_samples/golden_labels.example.json`; that file is only a synthetic safe-to-commit example. Use a local manually verified golden-label file for the same image.

Manual safety notes:
- Real Vision API calls may create cost.
- Do not run real Vision API commands unless intentionally testing.
- Do not commit `.env`, API keys, private BodyGraph images, raw base64 image data, private model responses, or private prediction files.
- Do not commit files under `data/bodygraph_samples/private/`.
- Mock and offline verification should work in a clean repo without OpenAI access.

### Future Phases

- Phase 3: RAG answer generation / reading generation using Phase 1 retrieval and Phase 2 `chart_data`.
- Phase 4: local app or API.
- Phase 5: cloud deployment / productionization.

Do not include implementation tasks for future-phase features in Phase 2.
