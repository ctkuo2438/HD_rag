"""Focused Q&A orchestration with lazy, injectable storage/provider boundaries."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from human_design.rag.bm25 import load_bm25_retriever
from human_design.rag.config import AppConfig, load_config
from human_design.rag.hybrid_index import HybridIndexError, load_manifest, load_nodes, require_real_embeddings
from human_design.rag.hybrid_retriever import HybridRetriever, Retriever
from human_design.rag.reranker import Reranker, create_reranker
from human_design.rag.retriever import load_dense_retriever
from human_design.reading.chart_context import InvalidChartError, build_chart_context
from human_design.reading.generator import generate_answer
from human_design.reading.models import (
    AnswerResult, AnswerStatus, ChartContext, ChartImageQuestionRequest, ChartQuestionRequest,
    KnowledgeQuestionRequest, PromptContext,
)
from human_design.reading.prompt import build_prompt_context
from human_design.reading.query_builder import build_retrieval_queries, validate_query
from human_design.vision.config import load_vision_config
from human_design.vision.models import BodyGraphExtractionResult
from human_design.vision.pipeline import extract_bodygraph


class ReadingPipelineError(ValueError):
    """Safe operational error; never includes private input or provider payloads."""


_BROAD_ENGLISH = re.compile(
    r"(?:please )?(?:"
    r"give me a (?:complete|full) reading(?: of my chart)?|"
    r"read my (?:entire|whole|complete) chart|"
    r"explain everything in my chart|"
    r"tell me everything about my (?:human design|chart)"
    r")(?: please)?"
)
_BROAD_CHINESE = re.compile(
    r"(?:請|请)?(?:完整解[讀读]我的人[類类][圖图]|[幫帮]我看完整[張张][圖图]|全部都解[釋释][給给]我)"
)


def _needs_focus(query: str) -> bool:
    """Match explicit whole-reading requests, not isolated complete/everything words."""
    normalized = re.sub(r"[\s,.!?;:，。！？；：、]+", " ", query).strip().casefold()
    return bool(_BROAD_ENGLISH.fullmatch(normalized)
                or _BROAD_CHINESE.fullmatch(normalized.replace(" ", "")))


def _focus_result() -> AnswerResult:
    return AnswerResult(AnswerStatus.NEEDS_FOCUS,
        "Please choose a specific topic, such as Authority, Profile, a Gate, a Channel, a Center, Type, Strategy, or Definition.")


def _extract_chart(image_path: Path) -> BodyGraphExtractionResult:
    try:
        config = load_vision_config()
    except Exception:
        raise ReadingPipelineError("Check HD_VISION_MODEL, HD_VISION_REASONING_EFFORT and HD_VISION_REAL_API") from None
    if config.real_api_enabled is not True:
        raise ReadingPipelineError("Image questions require HD_VISION_REAL_API=1")
    if not config.openai_api_key:
        raise ReadingPipelineError("Image questions require OPENAI_API_KEY")
    try:
        # Phase 2 already composes extraction/parser/interpreter/validation and
        # explicitly constructs BodyGraphExtractionResult. Reuse that flow intact.
        return extract_bodygraph(image_path=image_path, config=config)
    except Exception:
        raise ReadingPipelineError("BodyGraph extraction failed; check the image and Vision configuration") from None


def _insufficient_evidence() -> AnswerResult:
    return AnswerResult(AnswerStatus.INSUFFICIENT_EVIDENCE,
                        "There is insufficient retrieved evidence to answer this question.")


@dataclass(frozen=True)
class ReadingPipeline:
    """One service for knowledge, typed charts, and an outer Phase 2 image facade.

    Construction does not load config, indexes, or providers. Injected boundaries
    are trusted Python dependencies, never CLI flags or user request fields.
    """

    config: AppConfig | None = field(default=None, repr=False)
    dense_retriever: Retriever | None = field(default=None, repr=False)
    sparse_retriever: Retriever | None = field(default=None, repr=False)
    reranker: Reranker | None = field(default=None, repr=False)
    generator: Callable[[PromptContext, AppConfig], AnswerResult] | None = field(default=None, repr=False)
    extractor: Callable[[Path], BodyGraphExtractionResult] | None = field(default=None, repr=False)

    def answer_knowledge_question(self, request: KnowledgeQuestionRequest) -> AnswerResult:
        query = validate_query(request.query)
        if _needs_focus(query):
            return _focus_result()
        return self._answer(query, None)

    def answer_chart_question(self, request: ChartQuestionRequest) -> AnswerResult:
        query = validate_query(request.query)
        if _needs_focus(query):
            return _focus_result()
        try:
            context = build_chart_context(request.bodygraph_result)
        except InvalidChartError:
            return AnswerResult(AnswerStatus.INVALID_CHART,
                                "The chart failed validation. Please correct the chart extraction before asking again.")
        return self._answer(query, context)

    def answer_chart_image_question(self, request: ChartImageQuestionRequest) -> AnswerResult:
        query = validate_query(request.query)
        if _needs_focus(query):
            return _focus_result()
        if self.extractor is None:
            result = _extract_chart(request.bodygraph_image)
        else:
            try:
                result = self.extractor(request.bodygraph_image)
            except Exception:
                raise ReadingPipelineError("BodyGraph extraction failed") from None
        if not isinstance(result, BodyGraphExtractionResult):
            raise ReadingPipelineError("BodyGraph extraction must return BodyGraphExtractionResult")
        return self.answer_chart_question(ChartQuestionRequest(query, result))

    def _settings(self) -> AppConfig:
        if self.config is not None:
            return self.config
        try:
            return load_config()
        except ValueError as exc:
            # The existing config validator names fixed variables/constraints,
            # never environment values. Preserve its actionable opt-in guidance.
            raise ReadingPipelineError(str(exc)) from None
        except Exception:
            raise ReadingPipelineError(
                "Invalid reading configuration. Check HD_RAG settings, OPENAI_API_KEY and COHERE_API_KEY"
            ) from None

    def _retrievers(self, config: AppConfig) -> tuple[Retriever, Retriever]:
        if self.dense_retriever is not None or self.sparse_retriever is not None:
            if self.dense_retriever is None or self.sparse_retriever is None:
                raise HybridIndexError("Both dense and BM25 retrievers are required")
            return self.dense_retriever, self.sparse_retriever
        require_real_embeddings(config)
        try:
            manifest = load_manifest(config.index_dir / "manifest.json")
            nodes = load_nodes(config.index_dir / "nodes.jsonl", manifest)
            sparse = load_bm25_retriever(config.index_dir / "bm25", manifest.ingestion_id)
            if sparse.chunk_ids != tuple(node.node_id for node in nodes):
                raise HybridIndexError("BM25 canonical IDs do not match")
            dense = load_dense_retriever(config)
            if dense.ingestion_id != manifest.ingestion_id:
                raise HybridIndexError("Dense identity does not match")
            return dense, sparse
        except Exception:
            raise HybridIndexError(
                "Cannot load matching Phase 3 indexes; check HD_RAG_INDEX_DIR and rebuild the hybrid index"
            ) from None

    def _answer(self, query: str, chart: ChartContext | None) -> AnswerResult:
        request, selected = build_retrieval_queries(query, chart)
        config = self._settings()
        dense, sparse = self._retrievers(config)
        result = HybridRetriever(dense, sparse, config).retrieve(request)
        if not result.candidates:
            return _insufficient_evidence()
        try:
            reranker = self.reranker if self.reranker is not None else create_reranker(config)
            final = reranker.rerank(request.dense_query, result.candidates, config.final_top_k)[:config.final_top_k]
        except Exception:
            raise ReadingPipelineError(
                "Reranking failed; check HD_RAG_RERANK_PROVIDER, HD_RAG_REAL_RERANK_API, "
                "HD_RAG_RERANK_MODEL and COHERE_API_KEY; optional Cohere requires uv sync --extra rerank"
            ) from None
        if not final:
            return _insufficient_evidence()
        prompt = build_prompt_context(request.original_query, selected, final)
        try:
            generator = self.generator if self.generator is not None else generate_answer
            return generator(prompt, config)
        except Exception:
            raise ReadingPipelineError(
                "Generation failed or returned invalid evidence references; check HD_RAG_REAL_GENERATION=1, "
                "HD_RAG_GENERATION_MODEL and OPENAI_API_KEY"
            ) from None
