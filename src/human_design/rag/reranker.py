"""Minimal optional reranking over existing canonical chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from human_design.rag.config import DEFAULT_FINAL_TOP_K, AppConfig
from human_design.rag.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int,
    ) -> list[RetrievedChunk]: ...


def _limit(top_n: int, final_top_k: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (top_n, final_top_k)):
        raise ValueError("top_n and final_top_k must be positive integers")
    return min(top_n, final_top_k)


@dataclass(frozen=True)
class NoOpReranker:
    final_top_k: int = DEFAULT_FINAL_TOP_K

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return list(chunks[:_limit(top_n, self.final_top_k)])


def _require_cohere(config: AppConfig) -> None:
    if config.rerank_provider != "cohere":
        raise ValueError("Cohere requires HD_RAG_RERANK_PROVIDER=cohere")
    if config.real_rerank_api is not True:
        raise ValueError("Cohere requires HD_RAG_REAL_RERANK_API=1")
    if not isinstance(config.rerank_model, str) or not config.rerank_model.strip():
        raise ValueError("Cohere requires HD_RAG_RERANK_MODEL")
    if not isinstance(config.cohere_api_key, str) or not config.cohere_api_key.strip():
        raise ValueError("Cohere requires COHERE_API_KEY")
    _limit(config.final_top_k, config.final_top_k)


@dataclass(frozen=True)
class CohereReranker:
    config: AppConfig = field(repr=False)
    client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_cohere(self.config)
        if self.client is None:
            try:
                # The approved LlamaIndex rerank extra supplies this SDK. Using
                # it directly keeps document metadata and callbacks out of requests.
                from cohere import ClientV2
            except ImportError:
                raise ImportError("Optional Cohere support is unavailable; run uv sync --extra rerank") from None
            try:
                object.__setattr__(self, "client", ClientV2(api_key=self.config.cohere_api_key))
            except Exception:
                raise ValueError("Cannot construct Cohere reranker; check provider configuration") from None

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Use the semantic dense query; map provider order back to unchanged provenance."""
        _require_cohere(self.config)
        limit = min(_limit(top_n, self.config.final_top_k), len(chunks))
        if not chunks:
            return []
        try:
            response = self.client.rerank(
                model=self.config.rerank_model, query=query,
                documents=[chunk.text for chunk in chunks], top_n=limit,
            )
        except Exception:
            raise ValueError("Cohere reranking request failed") from None
        selected, seen = [], set()
        try:
            results = response.results
            for result in results:
                index = result.index
                if type(index) is not int or not 0 <= index < len(chunks) or index in seen:
                    raise ValueError("invalid index")
                seen.add(index)
                selected.append(replace(chunks[index], rerank_score=result.relevance_score))
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Cohere returned invalid result indexes or scores") from None
        return selected[:limit]


def create_reranker(config: AppConfig, *, client: Any = None) -> Reranker:
    """Select the configured boundary; provider none never imports the optional SDK."""
    if config.rerank_provider == "none":
        return NoOpReranker(config.final_top_k)
    if config.rerank_provider == "cohere":
        return CohereReranker(config, client=client)
    raise ValueError("HD_RAG_RERANK_PROVIDER must be none or cohere")
