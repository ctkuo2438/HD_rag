"""Small local web interface over the existing focused reading pipeline."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from streamlit.typing import UploadedFile

from human_design.rag.hybrid_index import HybridIndexError
from human_design.reading import (
    ChartImageQuestionRequest, ChartQuestionRequest, KnowledgeQuestionRequest, ReadingPipeline,
)
from human_design.reading.models import AnswerResult, AnswerStatus
from human_design.reading.pipeline import ReadingPipelineError
from human_design.reading.query_builder import InvalidQuestionError, MAX_QUERY_CHARACTERS, validate_query
from human_design.vision.models import BodyGraphExtractionResult


_IMAGE_TYPES = ("png", "jpg", "jpeg", "webp", "gif")
_MAX_UPLOAD_MB = 20
_FACT_LABELS = {
    "type": "類型", "authority": "內在權威", "strategy": "策略",
    "profile": "人生角色", "definition": "定義", "not_self_theme": "非自己主題",
    "signature": "標誌", "active_gate": "啟動閘門", "active_channel": "啟動通道",
    "defined_center": "已定義中心", "undefined_center": "未定義中心",
}


class InvalidUploadError(ValueError):
    """Safe upload validation message, without the original filename or bytes."""


def _answer(query: str, upload: UploadedFile | None) -> AnswerResult:
    query = validate_query(query)
    if upload is None:
        return ReadingPipeline().answer_knowledge_question(KnowledgeQuestionRequest(query))

    suffix = Path(upload.name).suffix.lower()
    if suffix.lstrip(".") not in _IMAGE_TYPES:
        raise InvalidUploadError("請上傳 PNG、JPEG、WebP 或 GIF 圖片。")
    if not 0 < upload.size <= _MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidUploadError("圖片不可為空，大小上限為 20 MB。")

    image_data = upload.getvalue()
    image_digest = sha256(image_data).hexdigest()
    cached = st.session_state.get("_validated_charts", {}).get(image_digest)
    if cached is not None:
        return ReadingPipeline().answer_chart_question(ChartQuestionRequest(query, cached))

    def remember_chart(result: BodyGraphExtractionResult) -> None:
        # Session-local typed data only: no shared cache, image bytes, or paths.
        # Save before answering so later retrieval/generation failures keep it.
        charts = st.session_state.get("_validated_charts", {})
        charts[image_digest] = result
        st.session_state["_validated_charts"] = charts

    # Never use the uploaded name as a destination. The private temporary copy
    # exists only for the image facade call and is removed even on failure.
    with TemporaryDirectory(prefix="hd-bodygraph-") as directory:
        image_path = Path(directory) / ("bodygraph" + suffix)
        image_path.write_bytes(image_data)
        return ReadingPipeline().answer_chart_image_question(
            ChartImageQuestionRequest(query, image_path), on_validated_chart=remember_chart)


def _show_answer(answer: AnswerResult) -> None:
    if answer.status is AnswerStatus.OK:
        st.markdown(answer.answer_markdown, unsafe_allow_html=False)
    elif answer.status in (AnswerStatus.ERROR, AnswerStatus.REFUSED):
        st.warning(answer.answer_markdown)
    else:
        st.info(answer.answer_markdown)

    if answer.citations:
        st.subheader("參考來源")
        for citation in answer.citations:
            page = citation.page_label if citation.page_label is not None else citation.page_number
            location = f" · 頁碼 {page}" if page is not None else ""
            st.text(f"[{citation.citation_id}] {citation.source_file}{location}")
    if answer.chart_facts_used:
        with st.expander("這次使用的圖表資訊"):
            for fact in answer.chart_facts_used:
                st.text(f"{_FACT_LABELS.get(fact.field, fact.field)}：{fact.value}")
    if answer.warnings:
        with st.expander("限制與提醒"):
            for warning in answer.warnings:
                st.text(warning)


def main() -> None:
    st.set_page_config(page_title="人類圖問答", layout="centered")
    st.title("人類圖問答")
    st.caption("一次問一個主題：權威、人生角色、閘門、通道、中心、類型、策略或定義。")

    with st.form("question_form", enter_to_submit=False):
        query = st.text_area(
            "你的問題", key="query", max_chars=MAX_QUERY_CHARACTERS,
            placeholder="例如：我該如何做決定？或：什麼是薦骨權威？",
        )
        upload = st.file_uploader(
            "人類圖圖片（選填）", type=list(_IMAGE_TYPES), key="bodygraph",
            max_upload_size=_MAX_UPLOAD_MB,
            help="上傳圖表可針對你的圖回答；未上傳時回答一般人類圖知識。單張圖片上限 20 MB。",
        )
        st.caption(
            "送出後會依本機設定呼叫 OpenAI，並可能使用 Cohere，會產生 API 費用。"
            "問題、參考段落與必要圖表資訊會送至服務商；圖片首次解析時會送至 OpenAI。"
            "同一工作階段內，同圖後續提問會重用已驗證的圖表資料。"
        )
        submitted = st.form_submit_button("送出問題", type="primary", key="submit")

    if submitted:
        # Clear the old answer before a new attempt, including invalid inputs.
        st.session_state.pop("answer", None)
        st.session_state.pop("answered_query", None)
        st.session_state.pop("answer_error", None)
        try:
            with st.spinner("正在處理問題，圖表解析可能需要幾分鐘…", show_time=True):
                answer = _answer(query, upload)
            st.session_state["answer"] = answer
            st.session_state["answered_query"] = query.strip()
        except InvalidQuestionError:
            st.session_state["answer_error"] = "請輸入 1 至 2,000 個字的問題，不可只有空白。"
        except (InvalidUploadError, HybridIndexError, ReadingPipelineError) as exc:
            # These boundaries own sanitized messages with actionable variable names.
            st.session_state["answer_error"] = str(exc)
        except Exception:
            # Do not hand private provider exceptions or local paths to Streamlit.
            st.session_state["answer_error"] = "暫時無法完成回答，請檢查本機設定與服務狀態後再試。"

    if error := st.session_state.get("answer_error"):
        st.error(error)
    if answer := st.session_state.get("answer"):
        st.divider()
        st.subheader("最近一次回答")
        st.text(st.session_state["answered_query"])
        _show_answer(answer)

    st.caption("人類圖資訊供反思與實驗，不替代醫療、法律或財務專業判斷。")


if __name__ == "__main__":
    main()
