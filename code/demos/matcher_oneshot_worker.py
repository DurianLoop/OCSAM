from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcher-root", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--box", required=True)
    parser.add_argument("--mask", default="")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matcher_root = Path(args.matcher_root)
    sys.path.insert(0, str(matcher_root))
    os.chdir(matcher_root)

    from gradio_demo.oss_ops_inference import main_oss_ops
    from segment_anything import sam_model_registry
    from dinov2.models import vision_transformer as vits
    import dinov2.utils.utils as dinov2_utils

    models_root = matcher_root / "models"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["default"](checkpoint=str(models_root / "sam_vit_h_4b8939.pth"))
    sam.to(device=device)
    sam.eval()

    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2 = vits.__dict__["vit_large"](**dinov2_kwargs)
    dinov2_utils.load_pretrained_weights(
        dinov2,
        str(models_root / "dinov2_vitl14_pretrain.pth"),
        "teacher",
    )
    dinov2.eval()
    dinov2.to(device=device)

    support = np.asarray(Image.open(args.support).convert("RGB"))
    target = np.asarray(Image.open(args.target).convert("RGB"))
    h, w = support.shape[:2]
    if args.mask:
        support_mask = np.asarray(Image.open(args.mask).convert("L"))
        if support_mask.shape != (h, w):
            support_mask = np.asarray(Image.fromarray(support_mask).resize((w, h), Image.Resampling.NEAREST))
        support_mask = (support_mask > 0).astype(np.uint8)
    else:
        x0, y0, x1, y1 = [int(round(float(v))) for v in args.box.split(",")]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        support_mask = np.zeros((h, w), dtype=np.uint8)
        support_mask[min(y0, y1) : max(y0, y1) + 1, min(x0, x1) : max(x0, x1) + 1] = 1

    old_argv = sys.argv[:]
    try:
        sys.argv = ["matcher_worker"]
        pred_masks, pred_mask_lists = main_oss_ops(
            sam=sam,
            dinov2=dinov2,
            support_img=support,
            support_mask=support_mask[None, ...],
            query_img_1=target,
            query_img_2=target,
            version=max(1, min(int(args.version), 3)),
        )
    finally:
        sys.argv = old_argv

    if args.version == 1 and pred_mask_lists[1] is not None:
        mask = pred_mask_lists[1][: min(9, len(pred_mask_lists[1]))].sum(0) > 0
    else:
        mask = pred_masks[1] > 0
    np.save(args.out, np.asarray(mask, dtype=bool))


if __name__ == "__main__":
    main()
