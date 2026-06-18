"""SAM3 adapter capability declaration."""

from __future__ import annotations

from typing import Any

from adapters.base import BasePromptAdapter, ModelSupport, Prediction, Prompt


class Sam3Adapter(BasePromptAdapter):
    name = "sam3"

    def load_model(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError("SAM3 runtime is currently hosted by demos.medical_sam_click_app.Sam3TextSegmenter")

    def predict(self, handle: Any, sample: Any, prompt: Prompt, options: dict[str, Any]) -> Prediction:
        raise NotImplementedError("Use PromptSamServer.segment while the phase-1 demo is being split.")

    def supports(self) -> ModelSupport:
        return ModelSupport(image=True, video=False, point=True, box=True, text=True, reference=False, native_text=True)
