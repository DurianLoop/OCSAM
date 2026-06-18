"""SAM2 adapter capability declaration."""

from __future__ import annotations

from typing import Any

from adapters.base import BasePromptAdapter, ModelSupport, Prediction, Prompt


class Sam2Adapter(BasePromptAdapter):
    name = "sam2"

    def load_model(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError("SAM2 runtime is currently hosted by demos.medical_sam_click_app.PromptSamServer")

    def predict(self, handle: Any, sample: Any, prompt: Prompt, options: dict[str, Any]) -> Prediction:
        raise NotImplementedError("Use PromptSamServer.segment or segment_sam2_video while the phase-1 demo is being split.")

    def supports(self) -> ModelSupport:
        return ModelSupport(image=True, video=True, point=True, box=True, text=False, reference=False)
