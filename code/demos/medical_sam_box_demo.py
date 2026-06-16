"""Small medical-image SAM demo using local checkpoints and datasets.

This script is intentionally lightweight: it uses the repository's bundled
`segment_anything` package plus the local SAM ViT-B checkpoint, runs a
box-prompt segmentation on a few medical samples, computes IoU/Dice against
available masks, and writes an HTML report.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry


def read_rgb(path: Path, max_side: int) -> tuple[np.ndarray, float]:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    return np.asarray(image), scale


def read_mask(path: Path, scale: float, max_side: int) -> np.ndarray:
    mask = Image.open(path)
    if scale < 1.0:
        w, h = mask.size
        mask = mask.resize((round(w * scale), round(h * scale)), Image.NEAREST)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return arr > 0


def mask_to_box(mask: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask cannot provide a prompt box")
    h, w = mask.shape
    return np.array(
        [
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(w - 1, xs.max() + pad),
            min(h - 1, ys.max() + pad),
        ],
        dtype=np.float32,
    )


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    iou = float(inter / union) if union else 1.0
    dice = float(2 * inter / denom) if denom else 1.0
    return iou, dice


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    out[mask] = 0.55 * out[mask] + 0.45 * tint[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    box: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    image_box = image.copy()
    x0, y0, x1, y1 = box.astype(int)
    boxed = Image.fromarray(image_box)
    draw = ImageDraw.Draw(boxed)
    for offset in range(2):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 210, 0))
    image_box = np.asarray(boxed)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [
        (image_box, "Image + prompt box"),
        (overlay(image, gt, (46, 160, 67)), "Ground truth"),
        (overlay(image, pred, (31, 119, 180)), "SAM prediction"),
        (overlay(overlay(image, gt, (46, 160, 67)), pred, (31, 119, 180)), "GT + prediction"),
    ]
    for ax, (panel, label) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def discover_samples(repo: Path) -> list[dict[str, Path | str]]:
    samples: list[dict[str, Path | str]] = []

    monuseg_img_dir = repo / "datasets" / "MonuSeg" / "kmms" / "images"
    monuseg_mask_dir = repo / "datasets" / "MonuSeg" / "kmms" / "masks"
    for ext in ("*.png", "*.tif", "*.tiff"):
        for image_path in sorted(monuseg_img_dir.glob(ext)):
            mask_path = None
            for mask_ext in (image_path.suffix, ".png", ".tif", ".tiff"):
                candidate = monuseg_mask_dir / f"{image_path.stem}{mask_ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is not None:
                samples.append(
                    {
                        "dataset": "MonuSeg nuclei",
                        "split": "kmms",
                        "image": image_path,
                        "mask": mask_path,
                    }
                )

    idrid_root = repo / "datasets" / "IDRID"
    for split_name, split_label in (
        ("DR_Training_Set", "train"),
        ("DR_Testing_Set", "test"),
    ):
        idrid_img_dir = idrid_root / split_name / "Fundus Images"
        idrid_mask_dir = idrid_root / split_name / "Combined Masks"
        for image_path in sorted(idrid_img_dir.glob("*.jpg")):
            mask_path = idrid_mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                samples.append(
                    {
                        "dataset": "IDRID diabetic retinopathy lesions",
                        "split": split_label,
                        "image": image_path,
                        "mask": mask_path,
                    }
                )

    return samples


def write_html(rows: Iterable[dict[str, str | float]], out_path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <section class="case">
              <h2>{html.escape(str(row["dataset"]))}: {html.escape(str(row["name"]))}</h2>
              <p>IoU: <strong>{row["iou"]:.3f}</strong> · Dice: <strong>{row["dice"]:.3f}</strong></p>
              <img src="{html.escape(str(row["panel"]))}" alt="SAM segmentation panel">
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Medical SAM Box-Prompt Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .note {{ color: #5f6368; line-height: 1.45; margin-bottom: 24px; }}
    .case {{ background: white; border: 1px solid #dfe3e8; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .case h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .case img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
<main>
  <h1>Medical SAM Box-Prompt Demo</h1>
  <p class="note">使用仓库内的 Segment Anything 实现与本地 SAM ViT-B 权重。
  这里用 ground-truth mask 的外接框模拟人工 box prompt，因此它是入门级 prompt 分割复现，不是完整 Matcher one-shot pipeline。</p>
  {''.join(cards)}
</main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/sam_vit_b.pth")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--out-dir", default="demo_outputs/medical_sam_box")
    args = parser.parse_args()

    repo = REPO_ROOT
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=str(repo / args.checkpoint))
    sam.to(device=device)
    sam.eval()
    print(f"Using device: {device}")
    predictor = SamPredictor(sam)

    rows = []
    for sample in discover_samples(repo)[: args.limit]:
        image_path = Path(sample["image"])
        mask_path = Path(sample["mask"])
        image, scale = read_rgb(image_path, args.max_side)
        gt = read_mask(mask_path, scale, args.max_side)
        box = mask_to_box(gt)

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(scores))
        pred = masks[best].astype(bool)
        iou, dice = metrics(pred, gt)

        safe_name = f"{sample['dataset']}_{image_path.stem}".replace(" ", "_").replace("/", "_")
        panel_path = out_dir / f"{safe_name}.png"
        save_panel(image, gt, pred, box, panel_path, f"{sample['dataset']} / {image_path.name}")
        rows.append(
            {
                "dataset": str(sample["dataset"]),
                "name": image_path.name,
                "iou": iou,
                "dice": dice,
                "panel": panel_path.name,
            }
        )
        print(f"{sample['dataset']} {image_path.name}: IoU={iou:.3f}, Dice={dice:.3f}")

    if not rows:
        raise RuntimeError("No local medical image/mask pairs were found.")

    report_path = out_dir / "index.html"
    write_html(rows, report_path)
    print(f"Demo report written to: {report_path}")


if __name__ == "__main__":
    main()
