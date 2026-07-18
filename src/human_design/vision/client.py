"""
Safe provider(OpenAI Vision API) boundary for local BodyGraph Vision extraction.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import cast

from human_design.vision.config import VisionConfig
from human_design.vision.constants import PLANETARY_FIELDS
from human_design.vision.prompt import load_bodygraph_raw_extraction_prompt


class VisionClientError(RuntimeError):
    """Raised when BodyGraph raw extraction cannot be completed safely."""


_SUPPORTED_IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_API_TIMEOUT_SECONDS = 300.0
_API_MAX_RETRIES = 2


def load_mock_response(path: Path) -> str:
    if not path.is_file():
        raise VisionClientError(f"Mock response file not found: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VisionClientError(f"Could not read mock response file: {path}") from exc


# entry point for local BodyGraph raw extraction, either from a mock fixture or real API call
def extract_bodygraph_raw_json(
    *,
    image_path: Path, # ex: data/bodygraph_samples/images/andy.png
    config: VisionConfig, # model, reasoning_effort, real_api_enabled, openai_api_key
    mock_response_path: Path | None = None, # local testing
) -> str:
    # validate the image file exists
    _require_image_file(image_path)

    # if a mock response is provided, use it instead of calling the real API
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
        raise VisionClientError("Real API mode requires HD_VISION_REAL_API=1.")
    if not config.openai_api_key:
        raise VisionClientError("OPENAI_API_KEY is required when HD_VISION_REAL_API=1.")

    # lazy import the OpenAI SDK only when real API mode is enabled
    try:
        from openai import OpenAI
        from openai.types.responses import ResponseInputParam, ResponseTextConfigParam
        from openai.types.shared_params import Reasoning
    except ImportError as exc:
        raise VisionClientError(
            "The OpenAI SDK is unavailable. Install the project's existing "
            "dependencies before using real API mode."
        ) from exc

    prompt = load_bodygraph_raw_extraction_prompt()

    media_type = (mimetypes.guess_type(image_path.name)[0])
    if media_type not in _SUPPORTED_IMAGE_TYPES:
        raise VisionClientError(
            f"Unsupported image type for {image_path.name}: {media_type or 'unknown'}"
        )

    # VisionClientError (size limit) propagates out untouched
    try:
        if image_path.stat().st_size > _MAX_IMAGE_BYTES:
            raise VisionClientError(
                f"Image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)}MB limit: "
                f"{image_path}"
            )

        # read_bytes() -> bytes, b64encode -> base64 bytes, decode -> str
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise VisionClientError(f"Could not read image file: {image_path}") from exc

    # API stage: anything raised here is a provider/SDK failure
    try:
        client = OpenAI(
            api_key=config.openai_api_key,
            timeout=_API_TIMEOUT_SECONDS,
            max_retries=_API_MAX_RETRIES,
        )

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

        text_config = cast(
            ResponseTextConfigParam,
            {
                "format": {
                    "type": "json_schema",
                    "name": "bodygraph_raw_extraction",
                    "strict": True, # model's output must match the schema exactly
                    "schema": _bodygraph_raw_extraction_schema(),
                }
            },
        )

        response = client.responses.create(
            model=config.model,
            reasoning=reasoning,
            input=response_input,
            text=text_config,
            store=False,
        )
    except Exception as exc:
        raise VisionClientError(f"Vision API request failed: {type(exc).__name__}: {exc}") from exc

    # extract the output text from the response
    output_text = response.output_text
    
    # check if the output text is valid and not empty
    if not isinstance(output_text, str) or not output_text.strip():
        raise VisionClientError("Vision API returned no raw JSON text.")

    return output_text


def _require_image_file(image_path: Path) -> None:
    if not image_path.is_file(): # file exist -> is_file() returns True
        raise VisionClientError(f"Image file not found: {image_path}")


# define the JSON schema for the raw BodyGraph extraction output
def _bodygraph_raw_extraction_schema() -> dict[str, object]:
    activation_properties = {
        # define each planetary field as a string or null
        field_name: {"type": ["string", "null"]}
        for field_name in PLANETARY_FIELDS
    }
    activation_column = {
        "type": "object",
        "properties": activation_properties,
        "required": list(PLANETARY_FIELDS),
        "additionalProperties": False,
    }
    uncertain_confidence = {"type": "number"}

    top_level_fields = (
        "personality",
        "design",
        "visually_defined_centers",
        "visually_active_gates",
        "visible_colored_channels",
        "uncertain_items",
    )
    return {
        "type": "object",
        "properties": {
            "personality": activation_column,
            "design": activation_column,
            "visually_defined_centers": {
                "type": "array",
                "items": {"type": "string"},
            },
            "visually_active_gates": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "visible_colored_channels": {
                "type": "array",
                "items": {"type": "string"},
            },
            "uncertain_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_path": {"type": "string"},
                        "observed_value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "null"},
                            ]
                        },
                        "reason": {"type": "string"},
                        "confidence": uncertain_confidence,
                    },
                    "required": [
                        "field_path",
                        "observed_value",
                        "reason",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": list(top_level_fields),
        "additionalProperties": False,
    }


__all__ = [
    "VisionClientError",
    "extract_bodygraph_raw_json",
    "extract_bodygraph_raw_json_with_real_api",
    "load_mock_response",
]
