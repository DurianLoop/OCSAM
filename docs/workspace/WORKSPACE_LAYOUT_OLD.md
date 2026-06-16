# D:\SAM Workspace Layout

This workspace now contains both the pulled official source code and the local medical-image reproduction workspace.

## Source Code

- `Matcher_official_fresh/`
  - Fresh GitHub clone of `https://github.com/aim-uofa/Matcher.git`.
  - Latest checked commit during setup: `d59b6c7 update readme`.
  - This is the clean upstream code.

- `code_pre/`
  - Local working/reproduction copy based on Matcher.
  - Includes medical dataset adapters, local datasets, checkpoints, demo scripts, and generated demo outputs.
  - Main demo created here: `code_pre/demos/medical_sam_box_demo.py`.

## Model Weights

Shared model directory:

- `pretrained_models/`

The same model files are also available under:

- `Matcher_official_fresh/models/`
- `code_pre/models/`

Current model files:

| File | Purpose |
| --- | --- |
| `dinov2_vitl14_pretrain.pth` | DINOv2 ViT-L feature extractor used by Matcher |
| `sam_vit_h_4b8939.pth` | SAM ViT-H checkpoint required by the official Matcher demo |
| `swint_only_sam_many2many.pth` | Semantic-SAM checkpoint used for the Semantic-SAM branch |
| `sam_vit_b.pth` | Smaller SAM ViT-B checkpoint used by the lightweight medical demo |
| `best_backbone_idrid_finetuned.pth` | Local IDRID-oriented DINOv2/backbone checkpoint |

The files in `pretrained_models/` and `Matcher_official_fresh/models/` were created from the existing `code_pre/models/` files, using hard links where Windows allowed it. This avoids duplicating several GB of model data.

## Re-run The Current Medical Demo

```powershell
cd D:\SAM\code_pre
python demos\medical_sam_box_demo.py --limit 4 --max-side 512
```

Output:

- `code_pre/demo_outputs/medical_sam_box/index.html`
- `code_pre/demo_outputs/medical_sam_box/*.png`

## Conda Environment

- Environment path: `D:\SAM\conda_envs\sam`
- Activation helper: `D:\SAM\activate_sam_env.ps1`
- Details: `D:\SAM\ENVIRONMENT.md`

## Notes

- `Matcher_official_fresh/` is the clean upstream source.
- `code_pre/` is the practical reproduction workspace with medical additions.
- `materials/` contains the paper PDFs for SAM, SAM2, SAM3, Matcher, MedSAM-Agent, One-Prompt, UM-SAM, OSAM-Fundus, and related work.
