"""Shared prompt and prediction contracts for workbench model adapters.

The phase-1 web app still calls the existing local model wrappers directly, but
these dataclasses define the stable contract that SAM, MedSAM, SAM2, Matcher,
and SAM3 adapters should converge on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PromptType = Literal[
    "point",
    "box",
    "text",
    "reference",
    "mask",
    "video_point",
    "video_box",
    "video_text",
]


@dataclass(slots=True)
class Prompt:
    type: PromptType
    points: list[tuple[float, float, int]] | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    text: str | None = None
    reference_image: Any | None = None
    reference_mask: Any | None = None
    frame_index: int | None = None


@dataclass(slots=True)
class Prediction:
    masks: list[Any] = field(default_factory=list)
    boxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    overlay: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSupport:
    image: bool
    video: bool
    point: bool
    box: bool
    text: bool
    reference: bool
    mask: bool = False
    native_text: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "image": self.image,
            "video": self.video,
            "point": self.point,
            "box": self.box,
            "text": self.text,
            "reference": self.reference,
            "mask": self.mask,
            "native_text": self.native_text,
        }


class BasePromptAdapter:
    name: str = "base"

    def load_model(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    def predict(self, handle: Any, sample: Any, prompt: Prompt, options: dict[str, Any]) -> Prediction:
        raise NotImplementedError

    def supports(self) -> ModelSupport:
        raise NotImplementedError
