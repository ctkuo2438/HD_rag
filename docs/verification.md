# Verification and feature map

Unit tests check each module in isolation, and passing them does not prove the app works for a user. This page lists the user-facing features, how a user reaches each one, what correct behavior looks like, and which automated check verifies it. When you add or change a user-facing feature, update its row and add a smoke check when the feature crosses modules.

## Offline smoke check

[scripts/smoke_offline.py](../scripts/smoke_offline.py) builds a synthetic PDF and a temporary hybrid index. It then runs scripted questions through real PDF extraction, chunking, Chroma, BM25, RRF, prompt rendering, answer validation and the Streamlit app. Only the embedding model, the answer model and Vision extraction are deterministic fakes. The fake answers use shapes real models produce, such as grouped citations and Markdown images. The corpus includes heading-only and page-number pages, as real books do.

The check never reads `.env`, production indexes or private charts, and calls no provider. Its output names statuses, error codes and citation IDs, never passages or prompts. Use the [offline environment](../README.md#default-verification) and replace `uv run pytest` with:

```sh
uv run python scripts/smoke_offline.py            # pipeline and UI checks
uv run python scripts/smoke_offline.py --browser  # also drive headless Chrome
```

The default test run already includes the pipeline and UI checks through `tests/test_smoke_offline.py`.

The browser checks are opt-in, because they need Chrome or Chromium on macOS or Linux; set `HD_SMOKE_CHROME` to use another executable. They run the real app in a local Streamlit server and drive it with headless Chrome. They also count requests to a local server that stands in for an attacker, after confirming the page can reach it. Screenshots are saved under `<work-dir>/screenshots`; the work directory is printed and kept. Run the browser checks after changing answer rendering, Markdown validation or the Streamlit app.

## Feature map

| ID | Feature | How a user reaches it | Correct behavior | Automated checks |
| --- | --- | --- | --- | --- |
| F1 | Knowledge question | Ask without uploading a chart | An answer with bracket citations; sources list file and page | smoke: pipeline; `test_reading_pipeline.py` |
| F2 | Grouped citations | The model writes `[S1, S2]` | Every grouped ID is validated and listed under 參考來源 | smoke: pipeline, UI, browser; `test_generation.py` |
| F3 | Short pages as sources | A heading-only or page-number page reaches the final sources | A normal answer, not a `private_output` failure | smoke: pipeline; `test_generation.py` |
| F4 | Links and images in answers | The model returns a Markdown link, image or URL | Rejected as `unsafe_markdown`; the browser fetches nothing | smoke: pipeline, UI, browser; `test_generation.py`, `test_phase3_models.py` |
| F5 | Unknown citations | The model cites an ID it was not given | Rejected as `unknown_citation` | smoke: pipeline; `test_generation.py` |
| F6 | Full-reading requests | Ask for a complete chart reading | `needs_focus` before any provider | smoke: pipeline; `test_reading_pipeline.py`, `test_streamlit_app.py` |
| F7 | Chart question | Upload a chart image and ask | The answer uses only the chart facts selected for the question | smoke: pipeline; `test_query_builder.py`, `test_reading_pipeline.py` |
| F8 | Underivable Authority | Ask about a chart whose Authority the rules cannot derive | No `Needs Review` fact in prompts or retrieval queries | smoke: pipeline; `test_chart_context.py` |
| F9 | Invalid chart | Upload a chart that fails validation | `invalid_chart` before retrieval or generation | smoke: pipeline; `test_reading_pipeline.py` |
| F10 | Same chart, next question | Ask again without changing the image | The validated chart is reused without a new extraction | smoke: UI; `test_streamlit_app.py` |
| F11 | Insufficient evidence | Retrieval returns no final sources | `insufficient_evidence` without generation | `test_reading_pipeline.py` |
| F12 | Hybrid index build and reload | Run `scripts/build_hybrid_index.py`, then ask | Matching Chroma and BM25 identity; queries never rebuild | `test_hybrid_index.py`, `test_bm25.py`; the smoke check also builds and reloads a real index |
| F13 | Chart extraction CLI | Run `scripts/extract_bodygraph.py` | Parsed, derived and validated chart JSON | `test_extract_bodygraph.py`; the [mock-response smoke command](development.md#tests-and-diagnostics) |
| F14 | Real providers | Enable paid opt-ins with real keys | Real embeddings, answers, Vision and optional Cohere work | Manual only, after explicit authorization |

## Not covered offline

- Whether a real model follows the prompt, for example by citing correctly or avoiding links. Validation rejects violations, but only a paid answer review shows how often they happen.
- Retrieval quality with real embeddings. Use `scripts/evaluate_hybrid_retrieval.py`, which incurs embedding cost.
- Vision accuracy on real charts. Use `scripts/evaluate_bodygraph_extraction.py` with manually verified labels.
- Visual layout beyond the two browser screenshots.
