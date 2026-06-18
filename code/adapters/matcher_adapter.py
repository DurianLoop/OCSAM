"""Matcher adapter capability declaration."""

from __future__ import annotations

from typing import Any

from adapters.base import BasePromptAdapter, ModelSupport, Prediction, Prompt


class MatcherAdapter(BasePromptAdapter):
    name = "matcher"

    def load_model(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError("Matcher runtime is currently hosted by demos.medical_sam_click_app.MatcherOneShotSegmenter")

    def predict(self, handle: Any, sample: Any, prompt: Prompt, options: dict[str, Any]) -> Prediction:
        raise NotImplementedError("Use PromptSamServer.segment_matcher while the phase-1 demo is being split.")

    def supports(self) -> ModelSupport:
        return ModelSupport(image=True, video=False, point=False, box=False, text=False, reference=True)
