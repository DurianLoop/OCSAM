"""Sample and scene registry for the phase-1 reproduction workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Difficulty = Literal["easy", "hard", "mixed"]


@dataclass(frozen=True, slots=True)
class SampleEntry:
    id: int
    name: str
    path: Path
    dataset: str
    scene: str
    difficulty: Difficulty
    media_type: Literal["image", "video", "frames"] = "image"

    def to_dict(self, url: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset": self.dataset,
            "scene": self.scene,
            "difficulty": self.difficulty,
            "media_type": self.media_type,
            "url": url,
        }


def _first(paths: list[Path], limit: int) -> list[Path]:
    return [p for p in paths if p.exists()][:limit]


def find_image_samples(repo_root: Path, workspace_root: Path) -> list[SampleEntry]:
    samples: list[SampleEntry] = []

    monuseg = _first(sorted((repo_root / "datasets" / "MonuSeg" / "kmms" / "images").glob("*.png")), 8)
    for idx, path in enumerate(monuseg):
        difficulty: Difficulty = "easy" if idx < 3 else "hard"
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="MonuSeg",
                scene="sparse nuclei" if difficulty == "easy" else "dense or touching nuclei",
                difficulty=difficulty,
            )
        )

    idrid = _first(
        sorted((repo_root / "datasets" / "IDRID" / "DR_Training_Set" / "Fundus Images").glob("*.jpg")),
        6,
    )
    for idx, path in enumerate(idrid):
        difficulty = "easy" if idx < 2 else "hard"
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="IDRID",
                scene="large bright lesion" if difficulty == "easy" else "small or low-contrast lesion",
                difficulty=difficulty,
            )
        )

    general = _first(sorted((workspace_root / "sam3" / "assets" / "images").glob("*.jpg")), 4)
    general.extend(_first(sorted((repo_root / "gradio_demo" / "images").glob("*.png")), 4))
    for path in general:
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="General",
                scene="open-vocabulary sanity check",
                difficulty="mixed",
            )
        )

    return samples


def find_video_samples(workspace_root: Path) -> list[SampleEntry]:
    paths: list[Path] = []
    paths.extend(sorted((workspace_root / "sam2" / "demo" / "data" / "gallery").glob("*.mp4")))
    bedroom = workspace_root / "sam2" / "notebooks" / "videos" / "bedroom"
    if bedroom.exists():
        paths.append(bedroom)
    paths.extend(sorted((workspace_root / "sam2" / "notebooks" / "videos").glob("*.mp4"))[:2])

    samples: list[SampleEntry] = []
    for idx, path in enumerate([p for p in paths if p.exists()]):
        media_type = "frames" if path.is_dir() else "video"
        samples.append(
            SampleEntry(
                id=idx,
                name=path.name,
                path=path,
                dataset="SAM2",
                scene="simple video propagation" if idx == 0 else "video propagation",
                difficulty="mixed",
                media_type=media_type,
            )
        )
    return samples


def sample_manifest_payload(image_samples: list[SampleEntry], video_samples: list[SampleEntry]) -> dict[str, Any]:
    return {
        "scenes": [
            {
                "dataset": "IDRID",
                "easy": "large bright lesion",
                "hard": "small or low-contrast lesion",
                "prompts": ["box", "point", "text: lesion/red lesion/bright lesion/exudate"],
            },
            {
                "dataset": "MonuSeg",
                "easy": "sparse nuclei",
                "hard": "dense or touching nuclei",
                "prompts": ["box", "point", "text: nucleus/cell nucleus/nuclei", "reference"],
            },
            {
                "dataset": "SAM2 video",
                "easy": "short stable object clip",
                "hard": "occlusion, appearance change, pseudo-video slices",
                "prompts": ["video_point", "video_box"],
            },
        ],
        "images": [sample.to_dict(url=f"/example/{sample.id}") for sample in image_samples],
        "videos": [
            {
                **sample.to_dict(url=f"/video/{sample.id}" if sample.path.is_file() else ""),
                "kind": "frames" if sample.path.is_dir() else "mp4",
            }
            for sample in video_samples
        ],
    }
