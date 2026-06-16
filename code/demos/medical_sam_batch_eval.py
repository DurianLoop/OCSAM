"""Batch evaluation for local medical SAM box-prompt examples."""

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

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from medical_sam_box_demo import (  # noqa: E402
    discover_samples,
    mask_to_box,
    metrics,
    read_mask,
    read_rgb,
    save_panel,
)
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


def filter_samples(samples: list[dict[str, Path | str]], dataset: str) -> list[dict[str, Path | str]]:
    if dataset == "all":
        return samples
    if dataset == "monuseg":
        return [s for s in samples if str(s["dataset"]).startswith("MonuSeg")]
    if dataset == "idrid":
        return [s for s in samples if str(s["dataset"]).startswith("IDRID")]
    if dataset == "idrid_train":
        return [s for s in samples if s.get("split") == "train"]
    if dataset == "idrid_test":
        return [s for s in samples if s.get("split") == "test"]
    raise ValueError(f"Unsupported dataset filter: {dataset}")


def summarize(rows: list[dict[str, str | float]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        groups.setdefault(f"{row['dataset']} / {row['split']}", []).append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for group, group_rows in groups.items():
        ious = [float(r["iou"]) for r in group_rows]
        dices = [float(r["dice"]) for r in group_rows]
        summary[group] = {
            "n": len(group_rows),
            "iou_mean": statistics.mean(ious),
            "iou_median": statistics.median(ious),
            "iou_min": min(ious),
            "iou_max": max(ious),
            "dice_mean": statistics.mean(dices),
            "dice_median": statistics.median(dices),
            "dice_min": min(dices),
            "dice_max": max(dices),
        }
    return summary


def write_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    fields = [
        "dataset",
        "split",
        "name",
        "status",
        "iou",
        "dice",
        "score",
        "elapsed_sec",
        "image",
        "mask",
        "panel",
        "error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict[str, str | float]], summary: dict[str, dict[str, float | int]], out_path: Path) -> None:
    summary_rows = []
    for group, item in summary.items():
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(group)}</td>
              <td>{item["n"]}</td>
              <td>{item["iou_mean"]:.3f}</td>
              <td>{item["iou_median"]:.3f}</td>
              <td>{item["dice_mean"]:.3f}</td>
              <td>{item["dice_median"]:.3f}</td>
            </tr>
            """
        )

    cards = []
    for row in rows:
        if row["status"] != "ok":
            continue
        panel = row.get("panel", "")
        panel_html = f'<img src="{html.escape(str(panel))}" alt="SAM segmentation panel">' if panel else ""
        cards.append(
            f"""
            <section class="case">
              <h2>{html.escape(str(row["dataset"]))} / {html.escape(str(row["split"]))}: {html.escape(str(row["name"]))}</h2>
              <p>IoU: <strong>{row["iou"]:.3f}</strong> | Dice: <strong>{row["dice"]:.3f}</strong> | Score: <strong>{row["score"]:.3f}</strong></p>
              {panel_html}
            </section>
            """
        )

    skipped = [r for r in rows if r["status"] != "ok"]
    skipped_rows = []
    for row in skipped:
        skipped_rows.append(
            f"<tr><td>{html.escape(str(row['name']))}</td><td>{html.escape(str(row['error']))}</td></tr>"
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Medical SAM Batch Evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .note {{ color: #5f6368; line-height: 1.45; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin: 18px 0 28px; }}
    th, td {{ border: 1px solid #dfe3e8; padding: 9px 10px; text-align: left; }}
    th {{ background: #eef2f6; }}
    .case {{ background: white; border: 1px solid #dfe3e8; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .case h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .case img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
<main>
  <h1>Medical SAM Batch Evaluation</h1>
  <p class="note">使用 ground-truth mask 的外接框模拟人工 box prompt，评估本地 SAM ViT-B 在已有医学图像样例上的 prompt segmentation 表现。</p>
  <table>
    <thead><tr><th>Dataset / split</th><th>N</th><th>Mean IoU</th><th>Median IoU</th><th>Mean Dice</th><th>Median Dice</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>
  <h2>Skipped samples</h2>
  <table>
    <thead><tr><th>Name</th><th>Reason</th></tr></thead>
    <tbody>{''.join(skipped_rows) if skipped_rows else '<tr><td colspan="2">None</td></tr>'}</tbody>
  </table>
  {''.join(cards)}
</main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def align_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    target_h, target_w = image.shape[:2]
    if mask.shape == (target_h, target_w):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((target_w, target_h), Image.NEAREST)
    return np.asarray(resized) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/sam_vit_b.pth")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 means all discovered samples")
    parser.add_argument(
        "--dataset",
        choices=["all", "monuseg", "idrid", "idrid_train", "idrid_test"],
        default="all",
    )
    parser.add_argument("--save-panels", choices=["all", "none"], default="all")
    parser.add_argument("--out-dir", default="demo_outputs/medical_sam_batch_eval")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = filter_samples(discover_samples(REPO_ROOT), args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No local medical image/mask pairs were found.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=str(REPO_ROOT / args.checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    print(f"Using device: {device}")
    print(f"Discovered samples: {len(samples)}")

    rows: list[dict[str, str | float]] = []
    total_start = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        image_path = Path(sample["image"])
        mask_path = Path(sample["mask"])
        try:
            image, scale = read_rgb(image_path, args.max_side)
            gt = read_mask(mask_path, scale, args.max_side)
            gt = align_mask_to_image(gt, image)
            box = mask_to_box(gt)
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            best = int(np.argmax(scores))
            pred = masks[best].astype(bool)
            iou, dice = metrics(pred, gt)
            elapsed = time.perf_counter() - start

            panel_name = ""
            if args.save_panels == "all":
                safe_name = (
                    f"{index:04d}_{sample['dataset']}_{sample['split']}_{image_path.stem}"
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                panel_path = out_dir / f"{safe_name}.png"
                save_panel(image, gt, pred, box, panel_path, f"{sample['dataset']} / {image_path.name}")
                panel_name = panel_path.name

            rows.append(
                {
                    "dataset": str(sample["dataset"]),
                    "split": str(sample["split"]),
                    "name": image_path.name,
                    "status": "ok",
                    "iou": iou,
                    "dice": dice,
                    "score": float(scores[best]),
                    "elapsed_sec": elapsed,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "panel": panel_name,
                    "error": "",
                }
            )
            print(
                f"[{index}/{len(samples)}] {sample['dataset']} / {sample['split']} {image_path.name}: "
                f"IoU={iou:.3f}, Dice={dice:.3f}, {elapsed:.2f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "dataset": str(sample["dataset"]),
                    "split": str(sample["split"]),
                    "name": image_path.name,
                    "status": "skipped",
                    "iou": "",
                    "dice": "",
                    "score": "",
                    "elapsed_sec": elapsed,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "panel": "",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(samples)}] skipped {image_path.name}: {exc}")

    summary = summarize(rows)
    payload = {
        "device": str(device),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "max_side": args.max_side,
        "dataset_filter": args.dataset,
        "total_samples": len(samples),
        "ok_samples": sum(1 for r in rows if r["status"] == "ok"),
        "skipped_samples": sum(1 for r in rows if r["status"] != "ok"),
        "elapsed_sec": time.perf_counter() - total_start,
        "summary": summary,
    }

    write_csv(rows, out_dir / "metrics.csv")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(rows, summary, out_dir / "index.html")

    print("Summary:")
    for group, item in summary.items():
        print(
            f"  {group}: n={item['n']}, "
            f"mean IoU={item['iou_mean']:.3f}, mean Dice={item['dice_mean']:.3f}"
        )
    print(f"CSV written to: {out_dir / 'metrics.csv'}")
    print(f"JSON written to: {out_dir / 'summary.json'}")
    print(f"HTML written to: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
