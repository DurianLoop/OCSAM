# SAM / Medical SAM Paper Reading Pack

This folder collects papers used to redesign Phase 2 before committing to a benchmark protocol.

Important boundary: the list intentionally mixes top-venue papers, high-impact foundation-model papers, and medical-SAM preprints. The goal is not to claim every file is a top-venue paper, but to learn how strong work designs experiments, defines prompts, handles domain gaps, and motivates innovation.

## Downloaded PDFs

| # | Paper | Local PDF | Source |
| --- | --- | --- | --- |
| 1 | Segment Anything | `01_Segment_Anything_SAM_ICCV2023.pdf` | https://arxiv.org/abs/2304.02643 |
| 2 | SAM 2: Segment Anything in Images and Videos | `02_SAM2_Segment_Anything_in_Images_and_Videos.pdf` | https://arxiv.org/abs/2408.00714 |
| 3 | SAM 3: Segment Anything with Concepts | `03_SAM3_Segment_Anything_with_Concepts.pdf` | https://arxiv.org/abs/2511.16719 |
| 4 | MedSAM: Segment Anything in Medical Images | `04_MedSAM_Segment_Anything_in_Medical_Images.pdf` | https://arxiv.org/abs/2304.12306 |
| 5 | Segment Anything in Medical Images and Videos: Benchmark and Deployment | `05_Segment_Anything_in_Medical_Images_and_Videos_Benchmark_Deployment.pdf` | https://arxiv.org/abs/2408.03322 |
| 6 | Medical SAM 2: Segment Medical Images as Video via Segment Anything Model 2 | `06_Medical_SAM2_Segment_Medical_Images_as_Video.pdf` | https://arxiv.org/abs/2408.00874 |
| 7 | MedSAM2: Segment Medical 3D Images and Videos | `07_MedSAM2_3D_Medical_Images_and_Videos.pdf` | https://arxiv.org/abs/2504.03600 |
| 8 | Computer-Vision Benchmark SAM in Medical Images: Accuracy in 12 Datasets | `08_Benchmark_SAM_in_Medical_Images_12_Datasets.pdf` | https://arxiv.org/abs/2304.09324 |
| 9 | Segment Anything Model for Medical Image Analysis: An Experimental Study | `09_SAM_for_Medical_Image_Analysis_Experimental_Study.pdf` | https://arxiv.org/abs/2304.10517 |
| 10 | Medical SAM Adapter | `10_Medical_SAM_Adapter.pdf` | https://arxiv.org/abs/2304.12620 |
| 11 | SAMed: Customized Segment Anything Model for Medical Image Segmentation | `11_SAMed_Customized_SAM_for_Medical_Image_Segmentation.pdf` | https://arxiv.org/abs/2304.13785 |
| 12 | MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation | `12_MA_SAM_Modality_Agnostic_SAM_Adaptation_3D.pdf` | https://arxiv.org/abs/2309.08842 |
| 13 | Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching | `13_Matcher_One_Shot_Feature_Matching.pdf` | https://arxiv.org/abs/2305.13310 |
| 14 | One-Prompt to Segment All Medical Images | `14_One_Prompt_to_Segment_All_Medical_Images.pdf` | https://arxiv.org/abs/2305.10300 |
| 15 | UniverSeg: Universal Medical Image Segmentation | `15_UniverSeg_Universal_Medical_Image_Segmentation.pdf` | https://arxiv.org/abs/2304.06131 |

Text extraction files are stored in `text/`. They are for local search and note-taking only.

## What To Learn From Each Paper

### 1. Segment Anything

Role in our project:

- Defines the original promptable segmentation protocol.
- Establishes model-size variants and the need to evaluate point, box, and mask prompts separately.

What matters for Phase 2:

- Do not evaluate SAM with only one prompt style.
- Report interaction cost, not only final mask quality.
- Keep SAM ViT-B as a speed baseline, but add stronger SAM variants if the goal is to compare against the best SAM-family baseline.

Design lesson:

```text
Benchmark by prompt type: 1-click, multi-click, box, automatic mask.
```

### 2. SAM 2

Role in our project:

- Defines image + video promptable segmentation.
- Introduces memory bank, object pointer, and interactive video correction.

What matters for Phase 2:

- Video evaluation cannot be a single "click once and inspect GIF" demo.
- We need first-frame prompt, correction-frame prompt, and temporal stability protocols.
- Online and offline interaction protocols should be separated.

Design lesson:

```text
For video, evaluate propagation and correction separately.
```

### 3. SAM 3

Role in our project:

- Moves the SAM family from low-level prompts toward concept-level/text-guided segmentation.
- Makes text prompt evaluation more important than the earlier SAM heuristic text mode.

What matters for Phase 2:

- Text prompt should be treated as a first-class protocol.
- Medical text prompts need templates, synonyms, and failure tags for semantic mismatch.
- We should not compare SAM3 only with box/click because that hides its central claim.

Design lesson:

```text
Text prompt benchmark must include vocabulary sensitivity and concept ambiguity.
```

### 4. MedSAM

Role in our project:

- The strongest immediate medical baseline in our current workbench.
- Shows that adapting SAM with large medical masks can beat raw zero-shot SAM on broad medical segmentation.

What matters for Phase 2:

- Box prompt is the primary fair protocol for MedSAM.
- Evaluation should cover modality breadth and external validation, not just a few curated examples.
- Robustness to box quality is a key clinical interaction issue.

Design lesson:

```text
MedSAM sets the box-prompt medical baseline; our innovation must beat or complement it.
```

### 5. Segment Anything in Medical Images and Videos

Role in our project:

- A bridge between medical images, videos, and deployment thinking.
- Useful for designing the transition from Phase 2 benchmark to Phase 3 prototype.

What matters for Phase 2:

- Medical "video" includes endoscopy, ultrasound, and 3D slice sequences.
- Deployment metrics should include runtime, memory, and interaction burden.

Design lesson:

```text
Benchmark must log resource cost and not only Dice.
```

### 6. Medical SAM 2

Role in our project:

- Treats 3D medical images as video-like sequences.
- Directly supports our proposed direction: cross-slice memory and SAM2-style propagation.

What matters for Phase 2:

- 2D slice metrics are insufficient for 3D tasks.
- We need slice-consistency and volume-consistency measurements.
- First-slice or sparse-slice prompting should be evaluated separately from dense prompting.

Design lesson:

```text
3D medical segmentation can be framed as promptable sequence segmentation.
```

### 7. MedSAM2

Role in our project:

- Extends the SAM2 idea to medical 3D images and videos.
- Confirms that memory/propagation is now a serious medical SAM direction.

What matters for Phase 2:

- Use it as a conceptual comparator for our later "memory selection" idea.
- Separate "can propagate" from "propagates accurately under anatomical change".

Design lesson:

```text
Measure memory usefulness, memory failure, and prompt correction efficiency.
```

### 8. SAM in Medical Images: 12-Dataset Benchmark

Role in our project:

- One of the most useful cautionary studies.
- Shows raw SAM underperforms medical specialist models and is sensitive to target size, contrast, modality, dimensionality, and difficulty.

What matters for Phase 2:

- Difficulty stratification is mandatory.
- Do not average all cases into one number.
- Record object size, contrast, modality, and 2D/3D status.

Design lesson:

```text
Every result table needs subgroup analysis.
```

### 9. SAM for Medical Image Analysis: Experimental Study

Role in our project:

- Another broad diagnostic study of raw SAM in medical settings.
- Useful for building failure taxonomy.

What matters for Phase 2:

- Raw SAM is not enough; prompt choice and target definition dominate performance.
- Manual, semi-automatic, and automatic settings should not be mixed.

Design lesson:

```text
Separate promptable segmentation, automatic segmentation, and zero-shot segmentation.
```

### 10. Medical SAM Adapter

Role in our project:

- Introduces an adapter-based way to adapt SAM to medical domains.
- Supports our planned "lightweight adapter" direction.

What matters for Phase 2:

- Adapter ablations matter: where to insert, how many parameters, what is frozen.
- Compare against full fine-tuning, prompt-only, and no-adaptation baselines.

Design lesson:

```text
If Phase 3 uses adapters, Phase 2 must collect the failure modes adapters should fix.
```

### 11. SAMed

Role in our project:

- Uses LoRA-style parameter-efficient fine-tuning for medical segmentation.
- Demonstrates a practical path to specialize SAM without training everything.

What matters for Phase 2:

- Add LoRA/adaptation as a future baseline lane.
- Evaluate parameter count and training cost, not only inference score.

Design lesson:

```text
Report adaptation cost as part of the benchmark.
```

### 12. MA-SAM

Role in our project:

- Modality-agnostic adaptation with 3D adapters.
- Important for CT/MRI/video medical data where pure 2D SAM loses context.

What matters for Phase 2:

- Include 2D-only vs 2.5D/3D context comparisons.
- Evaluate whether cross-slice context helps small/ambiguous structures.

Design lesson:

```text
A future innovation should be tested on both 2D slice quality and 3D consistency.
```

### 13. Matcher

Role in our project:

- In-context one-shot segmentation baseline.
- Gives an alternative to point/box/text prompts: reference image + reference mask.

What matters for Phase 2:

- Reference selection is a protocol, not a minor UI detail.
- Evaluate random support, nearest support, and oracle support.
- Version 1/2/3 should be mapped to multiple-instance, whole-object, and part-object behavior.

Design lesson:

```text
Benchmark support selection quality, not just Matcher output.
```

### 14. One-Prompt

Role in our project:

- Medical prompt automation direction.
- Relevant to reducing manual prompt effort.

What matters for Phase 2:

- Interaction efficiency should be measured as a primary endpoint.
- Prompt templates and learned prompts deserve ablation.

Design lesson:

```text
Prompt generation itself can be the innovation.
```

### 15. UniverSeg

Role in our project:

- Universal medical segmentation via support examples.
- Not a SAM model, but highly relevant for few-shot medical segmentation and reference-set evaluation.

What matters for Phase 2:

- Support-set size, support diversity, and held-out tasks are central.
- Few-shot medical segmentation should be evaluated with held-out task splits.

Design lesson:

```text
For reference-based segmentation, evaluate support size and support selection.
```

## Cross-Paper Conclusions

1. Stronger weights matter, but protocol matters more.
2. Raw SAM is not a reliable medical baseline unless prompt type, target size, modality, and difficulty are controlled.
3. MedSAM is the key box-prompt baseline.
4. SAM2/MedSAM2 shift the problem toward sequence memory and correction.
5. SAM3 shifts the problem toward text/concept prompt robustness.
6. Adapter and LoRA work suggests a feasible lightweight training direction.
7. Matcher and UniverSeg suggest that support/reference selection is a serious axis of innovation.
8. Phase 2 must become a diagnosis stage: it should identify which failure modes are worth solving in Phase 3.
