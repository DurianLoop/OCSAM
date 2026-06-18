"""Model capability registry for the phase-1 web workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.base import ModelSupport


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    key: str
    label: str
    role: str
    support: ModelSupport
    checkpoint: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "role": self.role,
            "support": self.support.to_dict(),
            "checkpoint": self.checkpoint,
            "notes": self.notes,
        }


def build_model_registry(workspace_root: Path) -> list[ModelRegistryEntry]:
    ckpt = workspace_root / "assets" / "checkpoints"
    return [
        ModelRegistryEntry(
            key="sam",
            label="SAM ViT-B",
            role="Generic promptable segmentation baseline",
            support=ModelSupport(image=True, video=False, point=True, box=True, text=True, reference=False),
            checkpoint=str(ckpt / "sam_vit_b.pth"),
            notes="Text mode is heuristic SAM automatic-mask selection, not native SAM text segmentation.",
        ),
        ModelRegistryEntry(
            key="medsam",
            label="MedSAM ViT-B",
            role="Medical box-prompt segmentation baseline",
            support=ModelSupport(image=True, video=False, point=False, box=True, text=False, reference=False),
            checkpoint=str(ckpt / "medsam_vit_b.pth"),
            notes="Box prompt only in this workbench.",
        ),
        ModelRegistryEntry(
            key="sam2",
            label="SAM2.1 Tiny",
            role="Image/video promptable segmentation baseline",
            support=ModelSupport(image=True, video=True, point=True, box=True, text=False, reference=False),
            checkpoint=str(ckpt / "sam2.1_hiera_tiny.pt"),
            notes="Video mode uses first-frame point or box prompts and propagates masks.",
        ),
        ModelRegistryEntry(
            key="matcher",
            label="Matcher",
            role="One-shot/reference segmentation baseline",
            support=ModelSupport(image=True, video=False, point=False, box=False, text=False, reference=True),
            checkpoint=str(ckpt / "swint_only_sam_many2many.pth"),
            notes="Reference image plus reference box support protocol.",
        ),
        ModelRegistryEntry(
            key="sam3",
            label="SAM3",
            role="Open-vocabulary concept segmentation baseline",
            support=ModelSupport(
                image=True,
                video=False,
                point=True,
                box=True,
                text=True,
                reference=False,
                native_text=True,
            ),
            checkpoint=str(ckpt / "sam3_modelscope" / "sam3.pt"),
            notes="Text mode is native concept prompting; click/box use local visual prompt integration.",
        ),
    ]


def registry_payload(workspace_root: Path) -> dict[str, Any]:
    entries = build_model_registry(workspace_root)
    return {"models": [entry.to_dict() for entry in entries]}
