# OCSAM Workspace Guide

OCSAM is a medical image segmentation reproduction workspace built around Matcher, SAM, DINOv2, and medical SAM-style evaluation scripts.

## What Is Included

| Path | Purpose |
|---|---|
| `demos/medical_sam_box_demo.py` | Small SAM box-prompt medical segmentation demo |
| `demos/medical_sam_batch_eval.py` | Batch evaluation on local MonuSeg and IDRID samples |
| `matcher/` | Matcher and medical-data adapters |
| `segment_anything/` | Local SAM implementation used by the demos |
| `semantic_sam/` | Semantic-SAM related modules from the original codebase |
| `docs/paper_code_map.md` | Chinese guide mapping SAM/MedSAM/Matcher papers to code |
| `UI/SAM_GUI.py` | Lightweight local GUI entry point |

Large checkpoints, local medical datasets, and generated outputs are intentionally not committed.

## Expected Local Layout

Place checkpoints under:

```text
D:\SAM\assets\checkpoints
```

In the local workspace, `models/` is a junction to that shared checkpoint folder. The scripts expect names such as:

```text
models/sam_vit_b.pth
models/sam_vit_h_4b8939.pth
models/dinov2_vitl14_pretrain.pth
```

## Run The GPU Environment

From the workspace root:

```powershell
cd D:\SAM
.\activate_sam_env.ps1
cd D:\SAM\code
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run The Current Batch Baseline

```powershell
python demos\medical_sam_batch_eval.py --dataset all --max-side 512 --out-dir demo_outputs\medical_sam_batch_eval_full_v2
```

This evaluates SAM ViT-B with ground-truth bounding-box prompts. It is a baseline, not the final Matcher one-shot pipeline.

Latest local baseline summary:

| Dataset / split | N | Mean IoU | Mean Dice |
|---|---:|---:|---:|
| MonuSeg nuclei / kmms | 82 | 0.308 | 0.443 |
| IDRID diabetic retinopathy lesions / train | 54 | 0.059 | 0.110 |
| IDRID diabetic retinopathy lesions / test | 27 | 0.063 | 0.118 |

## Next Steps

1. Add MedSAM or medical SAM checkpoints and run the same protocol.
2. Run Matcher one-shot segmentation with reference image/mask prompts.
3. Build a Gradio demo that switches between SAM baseline, MedSAM, and Matcher.
4. Keep datasets, outputs, and checkpoints outside Git; commit only code and lightweight docs.
