"""Project gateway text and cache usage into the shared native cost calculator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # native callable validation
)

from pydantic import Field, TypeAdapter

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteContext, decision
from litellm.rust_bridge.configuration import Decision

if TYPE_CHECKING:
    from litellm.litellm_core_utils.llm_cost_calc.utils import CompletionTokensDetailsResult, PromptTokensDetailsResult
    from litellm.types.utils import ModelInfo, Usage


class CatalogCostCalculator(Protocol):
    def __call__(self, pricing_json: bytes, request_json: bytes) -> object: ...


def _as_calculator(value: object) -> CatalogCostCalculator | None:
    return (
        cast(CatalogCostCalculator, value)  # cast-ok: native extension callable validated at the binding boundary
        if callable(value)
        else None
    )


CATALOG_COST: Final = NativeBinding("calculate_catalog_cost", validate=_as_calculator)
_Cost: TypeAlias = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_COST: Final[TypeAdapter[tuple[float, float] | None]] = TypeAdapter(tuple[_Cost, _Cost] | None)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CONTEXT: Final = RouteContext(Route.COST_CALCULATOR)
_RATE_PREFIXES: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "output_cost_per_image_token",
    "output_cost_per_reasoning_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)


@dataclass(frozen=True, slots=True)
class _Request:
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    reasoning_tokens: int
    service_tier: str | None
    billed_at_ns: int
    threshold_is_inclusive: bool


def calculate_cost(
    model_info: ModelInfo,
    custom_llm_provider: str,
    usage: Usage,
    prompt: PromptTokensDetailsResult,
    completion: CompletionTokensDetailsResult | None,
    service_tier: str | None,
    billed_at: datetime,
    threshold_is_inclusive: bool,
    *,
    selected: Decision | None = None,
) -> tuple[float, float] | None:
    if (decision(_CONTEXT) if selected is None else selected) is Decision.PYTHON:
        return None
    calculator: Final = CATALOG_COST.load()
    if calculator is None:
        return None
    if any(
        (
            prompt["cache_hit_audio_tokens"],
            prompt["audio_tokens"],
            prompt["image_tokens"],
            prompt["video_tokens"],
            prompt["character_count"],
            prompt["image_count"],
            prompt["video_length_seconds"],
            prompt["audio_length_seconds"],
            prompt["query_count"],
        )
    ) or (
        completion is not None
        and any((completion["audio_tokens"], completion["image_tokens"], completion["video_tokens"]))
    ):
        return None
    if prompt["text_tokens"] + prompt["cache_hit_tokens"] + prompt["cache_creation_tokens"] != usage.prompt_tokens:
        return None
    reasoning: Final = completion["reasoning_tokens"] if completion is not None else 0
    reported_text: Final = completion["text_tokens"] if completion is not None else 0
    text: Final = reported_text if reported_text != 0 else max(usage.completion_tokens - reasoning, 0)
    if text + reasoning != usage.completion_tokens:
        return None
    durations: Final = prompt["cache_creation_token_details"]
    five: Final = durations.ephemeral_5m_input_tokens or 0 if durations is not None else prompt["cache_creation_tokens"]
    one: Final = durations.ephemeral_1h_input_tokens or 0 if durations is not None else 0
    if five + one != prompt["cache_creation_tokens"]:
        return None
    reference: Final = billed_at.replace(tzinfo=timezone.utc) if billed_at.tzinfo is None else billed_at
    elapsed: Final = reference - _EPOCH
    request: Final = _Request(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cache_read_tokens=prompt["cache_hit_tokens"],
        cache_write_tokens=prompt["cache_creation_tokens"],
        cache_write_5m_tokens=five,
        cache_write_1h_tokens=one,
        reasoning_tokens=reasoning,
        service_tier=service_tier,
        billed_at_ns=((elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds) * 1000,
        threshold_is_inclusive=threshold_is_inclusive,
    )
    try:
        price_fields: Final = {
            key: value
            for key, value in model_info.items()
            if key.startswith(_RATE_PREFIXES) or key in ("tiered_pricing", "off_peak_pricing")
        }
        pricing_json: Final = json.dumps(
            {**price_fields, "litellm_provider": custom_llm_provider}, allow_nan=False
        ).encode()
        request_json: Final = json.dumps(asdict(request), allow_nan=False).encode()
    except (TypeError, ValueError):
        return None
    return _COST.validate_python(calculator(pricing_json, request_json))
