"""Safe provider boundary for local BodyGraph Vision extraction."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import cast

from human_design.vision.config import VisionConfig
from human_design.vision.prompt import load_bodygraph_raw_extraction_prompt


class VisionClientError(RuntimeError):
    """Raised when BodyGraph raw extraction cannot be completed safely."""


def load_mock_response(path: Path) -> str:
    """Read a sanitized local raw JSON response fixture."""
    if not path.is_file():
        raise VisionClientError(f"Mock response file not found: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VisionClientError(
            f"Could not read mock response file: {path}"
        ) from exc


def extract_bodygraph_raw_json(
    *,
    image_path: Path,
    config: VisionConfig,
    mock_response_path: Path | None = None,
) -> str:
    """Return raw extraction JSON from a mock fixture or explicit real API call."""
    _require_image_file(image_path)

    if mock_response_path is not None:
        return load_mock_response(mock_response_path)

    if not config.real_api_enabled:
        raise VisionClientError(
            "Provide a mock response or set HD_VISION_REAL_API=1 "
            "for real API mode."
        )

    if not config.openai_api_key:
        raise VisionClientError(
            "OPENAI_API_KEY is required when HD_VISION_REAL_API=1."
        )

    return extract_bodygraph_raw_json_with_real_api(
        image_path=image_path,
        config=config,
    )


def extract_bodygraph_raw_json_with_real_api(
    image_path: Path,
    config: VisionConfig,
) -> str:
    """Call the approved OpenAI Vision provider after explicit safety checks."""
    if not config.real_api_enabled:
        raise VisionClientError(
            "Real API mode requires HD_VISION_REAL_API=1."
        )

    if not config.openai_api_key:
        raise VisionClientError(
            "OPENAI_API_KEY is required when HD_VISION_REAL_API=1."
        )

    try:
        from openai import OpenAI
        from openai.types.responses import ResponseInputParam
        from openai.types.shared_params import Reasoning
    except ImportError as exc:
        raise VisionClientError(
            "The OpenAI SDK is unavailable. Install the project's existing "
            "dependencies before using real API mode."
        ) from exc

    prompt = load_bodygraph_raw_extraction_prompt()
    media_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "application/octet-stream"
    )

    try:
        encoded_image = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        client = OpenAI(api_key=config.openai_api_key)

        reasoning = cast(
            Reasoning,
            {
                "effort": config.reasoning_effort,
            },
        )

        response_input = cast(
            ResponseInputParam,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{media_type};base64,{encoded_image}"
                            ),
                        },
                    ],
                }
            ],
        )

        response = client.responses.create(
            model=config.model,
            reasoning=reasoning,
            input=response_input,
        )

    except OSError as exc:
        raise VisionClientError(
            f"Could not read image file: {image_path}"
        ) from exc
    except Exception as exc:
        raise VisionClientError(
            "Vision API request failed."
        ) from exc

    output_text = response.output_text

    if not isinstance(output_text, str) or not output_text.strip():
        raise VisionClientError(
            "Vision API returned no raw JSON text."
        )

    return output_text


def _require_image_file(image_path: Path) -> None:
    """Require the supplied local image path to reference an existing file."""
    if not image_path.is_file():
        raise VisionClientError(
            f"Image file not found: {image_path}"
        )


__all__ = [
    "VisionClientError",
    "extract_bodygraph_raw_json",
    "extract_bodygraph_raw_json_with_real_api",
    "load_mock_response",
]