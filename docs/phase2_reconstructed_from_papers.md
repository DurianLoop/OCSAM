# Phase 2 重构：论文驱动的医学 SAM Benchmark

## 0. 为什么要重构 Phase 2

原来的 Phase 2 目标是：

```text
每个模型在同一数据、同一 prompt 下跑表格，找谁在哪些场景崩。
```

这句话方向是对的，但还不够像一篇强论文的实验设计。读完 `docs/papers2` 这一组论文后，Phase 2 应该升级为：

```text
建立一个医学 promptable segmentation 诊断平台，
系统衡量模型、prompt、交互成本、医学场景难度、时序记忆和参考样本选择之间的关系，
从而为 Phase 3 的医学 prompt 自动化、SAM2 memory 选择和轻量 adapter 提供可证伪的实验依据。
```

也就是说，Phase 2 不只是跑 leaderboard，而是要回答：

1. 哪一种 prompt 对哪类医学目标最有效？
2. 强 checkpoint 是否真的解决医学目标的困难，还是只提高通用图像能力？
3. 失败来自模型、prompt、support/reference、时序传播，还是医学图像本身的低对比/小目标/多实例？
4. 我们 Phase 3 的创新到底该解决哪个最有价值的问题？

## 1. 新 Phase 2 总目标

Phase 2 的新目标是构建一个论文级 benchmark 和 failure diagnosis system。

核心接口仍然保持：

```python
predict(image_or_video, prompt, model_name, options) -> Prediction
```

但评价维度从单一 mask 分数扩展为：

```text
模型能力 = 准确率 + prompt 鲁棒性 + 交互效率 + 医学难度适应性 + 时序稳定性 + 参考样本敏感性
```

## 2. 从论文得到的实验设计原则

### 2.1 不要只比较模型，要比较 prompt 协议

来自 SAM/SAM2 的启发：

- 1-click、3-click、5-click、box、mask prompt 是不同任务。
- 视频中的 first-frame prompt 和 correction prompt 是不同任务。
- Text prompt 是 concept-level 任务，不能和 box prompt 简单混为一谈。

Phase 2 必须把 prompt 写进实验矩阵，而不是只写模型名。

### 2.2 不要只报告平均 Dice，要做医学难度分层

来自 12-dataset medical SAM benchmark 和 experimental study 的启发：

SAM 在医学图像中明显受这些因素影响：

- 目标大小。
- 前景-背景对比度。
- 图像模态。
- 2D vs 3D。
- 目标边界清晰度。
- 任务本身难度。

Phase 2 的每张图都要记录 difficulty metadata。

建议字段：

```csv
case_id,modality,target_area_ratio,contrast_score,component_count,boundary_complexity,difficulty_bucket
```

### 2.3 不要把所有 SAM 权重当成同一类 baseline

用户怀疑当前权重不是最强模型，这是合理的。

Phase 2 应该把模型分成三层：

```text
lightweight demo baseline:
  SAM ViT-B
  SAM2.1 Tiny

strong public baseline:
  SAM ViT-H
  SAM2.1 Large / Base+
  MedSAM ViT-B

research comparison baseline:
  SAM3
  MedSAM2 / Medical SAM2
  SAMed / Medical SAM Adapter / MA-SAM
  Matcher / UniverSeg-like reference methods
```

第一阶段网页可以继续用轻量模型保证交互流畅；第二阶段正式 benchmark 不能只用轻量模型。

### 2.4 MedSAM 是 box-prompt 医学主基线

来自 MedSAM：

- MedSAM 的核心场景是 box prompt。
- 它不是拿来证明 text/click 的。
- 它应该作为医学 box-prompt 的强 baseline。

Phase 2 中：

```text
box prompt protocol: SAM ViT-B/H vs MedSAM vs SAM2 image vs SAM3 optional box
```

### 2.5 SAM2/MedSAM2 的重点是 memory 和 correction

来自 SAM2、Medical SAM2、MedSAM2：

- 视频/3D 序列不能只看首帧结果。
- 需要看传播、漂移、消失、纠错效率。
- 3D 医学图像可以被看成 slice video。

Phase 2 必须新增：

```text
first-slice prompt
sparse-slice prompt
correction-slice prompt
memory failure tags
```

### 2.6 SAM3 的重点是 text/concept，不是再跑一个 box baseline

来自 SAM3：

- SAM3 的关键价值是 concept-level segmentation。
- 医学 text prompt 不能只写一个词，要做模板和同义词。

Phase 2 应该设计：

```text
simple text: lesion
medical text: retinal lesion
attribute text: small red retinal lesion
anatomical text: nuclei in histopathology image
negative or disambiguated text: lesion not vessel
```

### 2.7 Matcher/UniverSeg 的重点是 reference selection

来自 Matcher 和 UniverSeg：

- reference/support 不是随便选的。
- 支持样本的质量和数量会显著改变结果。

Phase 2 必须把 support protocol 当成一等公民：

```text
random support
same-dataset nearest support
oracle support
cross-dataset support
support-size ablation
```

## 3. 新 Phase 2 研究问题

### RQ1：强 SAM 权重是否足以解决医学图像？

比较：

- SAM ViT-B
- SAM ViT-H
- SAM2 Tiny
- SAM2 Large/Base+
- MedSAM

Prompt：

- box_tight
- box_loose_10
- click_1p
- click_3p

结论目标：

```text
判断“换更强权重”是否足够，还是必须医学适配。
```

### RQ2：医学场景中，哪种 prompt 最划算？

比较：

- click
- box
- text
- reference

指标：

- Dice
- IoU
- interaction cost
- time-to-mask
- prompt sensitivity

结论目标：

```text
找到不同医学任务的最佳 prompt 协议。
```

### RQ3：SAM3 的 text prompt 在医学语义上可靠吗？

比较：

- SAM3 text
- SAM heuristic text
- SAM/MedSAM box upper bound

Prompt 模板：

- target-only
- target + modality
- target + appearance
- target + anatomy

结论目标：

```text
判断医学文本 prompt 是否需要结构化模板和医学词表。
```

### RQ4：SAM2 memory 在医学序列中什么时候会失败？

任务：

- 视频。
- 3D slice sequence。
- 伪视频：将 CT/MRI 切片作为帧。

指标：

- frame Dice / slice Dice。
- temporal consistency。
- area jitter。
- centroid drift。
- correction clicks needed。

结论目标：

```text
为 Phase 3 的 memory selection 提供失败证据。
```

### RQ5：Reference-based 方法能否替代人工 prompt？

比较：

- Matcher v1/v2/v3。
- UniverSeg-style support protocol。
- SAM/MedSAM box prompt。

Support：

- random。
- nearest。
- oracle。
- cross-dataset。

结论目标：

```text
判断 reference retrieval 是否值得成为 Phase 3 的创新模块。
```

### RQ6：轻量 adapter 应该修复什么？

对照：

- raw SAM。
- MedSAM。
- SAMed/Medical SAM Adapter/MA-SAM 思路。

分析：

- 小目标失败。
- 低对比失败。
- 跨模态失败。
- 3D 不连续失败。

结论目标：

```text
确定 adapter 的训练目标和 ablation 维度。
```

## 4. 新实验矩阵

### 4.1 2D 医学图像主矩阵

| Dataset | Task | Difficulty | Prompt | Models |
| --- | --- | --- | --- | --- |
| IDRID | retinal lesion | small, low contrast | box/click/text | SAM-B, SAM-H, MedSAM, SAM2, SAM3 |
| MonuSeg | nuclei | dense multi-instance | box/click/text/reference | SAM-B, SAM-H, MedSAM, SAM2, SAM3, Matcher |

必做指标：

- Dice。
- IoU。
- Precision。
- Recall。
- Boundary F1。
- Runtime。
- Prompt count。

必做分层：

- small / medium / large target。
- low / medium / high contrast。
- sparse / dense instance。

### 4.2 Strong-weight sanity matrix

目标：

验证当前网页 demo 的轻量权重是否低估 SAM/SAM2。

模型：

```text
SAM ViT-B
SAM ViT-H
SAM2.1 Tiny
SAM2.1 Base+ or Large
MedSAM
```

协议：

```text
MonuSeg 20 cases
IDRID 20 cases
box_tight
click_1p
click_3p
```

输出：

```text
Does stronger checkpoint close the medical gap?
```

### 4.3 Text/concept matrix

目标：

专门评价 SAM3 和医学 text prompt。

Prompt levels：

```text
L1: lesion
L2: retinal lesion
L3: small red retinal lesion
L4: diabetic retinopathy lesion in fundus image
```

对照：

```text
SAM3 text
SAM heuristic text
box-prompt upper bound
```

失败标签：

- text_mismatch。
- semantic_overreach。
- misses_small_target。
- vessel_confusion。
- background_activation。

### 4.4 Video / 3D sequence matrix

目标：

评估 SAM2/Medical SAM2/MedSAM2 思路的 memory 行为。

协议：

```text
first_frame_click
first_frame_box
first_frame_mask_oracle
correction_at_worst_frame
sparse_slice_prompt_every_k
```

指标：

- mean frame Dice。
- worst frame Dice。
- area jitter。
- centroid drift。
- disappearance count。
- correction count to recover Dice >= 0.85。

### 4.5 Reference prompt matrix

目标：

评估 Matcher/UniverSeg 这类 reference-based 方法。

Support selection：

```text
random support
nearest support by image embedding
nearest support by mask shape
oracle support
cross-dataset support
```

Matcher version：

```text
v1 multiple instances
v2 whole object
v3 part object
```

指标：

- Dice。
- IoU。
- support sensitivity。
- best-minus-random gap。
- reference mismatch tags。

## 5. 新 Phase 2 交付物

### 5.1 Paper pack

位置：

```text
D:\SAM\docs\papers2
```

内容：

- PDF。
- 文本抽取。
- 逐篇学习笔记。

### 5.2 Benchmark manifest

位置：

```text
D:\SAM\runs\phase2_benchmark\manifests
```

必须包含：

```text
dataset,case_id,split,image_path,mask_path,task,modality,target_label,target_area_ratio,contrast_score,component_count,difficulty_bucket
```

### 5.3 Prompt manifest

位置：

```text
D:\SAM\runs\phase2_benchmark\prompts
```

必须包含：

```text
case_id,prompt_protocol,prompt_json,seed,source
```

### 5.4 Prediction artifacts

每个预测必须保存：

```text
mask.png
overlay.png
metadata.json
```

metadata 至少包含：

```json
{
  "model": "sam_vit_h",
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "prompt_protocol": "click_3p",
  "runtime_ms": 0,
  "device": "cuda",
  "status": "ok"
}
```

### 5.5 Failure gallery

必须输出一个 HTML 或 Markdown gallery，按失败类型分组：

- 小目标漏检。
- 背景过分割。
- 多实例合并。
- 文本语义错误。
- reference mismatch。
- memory drift。

## 6. Phase 2 阶段性路线图

### Milestone A：权重强度核查

目标：

```text
确认 SAM-B/SAM-H/SAM2-Tiny/SAM2-Large/MedSAM 的真实差距。
```

原因：

用户提出的怀疑是正确的：如果只用 SAM ViT-B 和 SAM2 Tiny，就不能代表最强 SAM-family baseline。

完成标准：

- SAM ViT-H 权重可用或明确记录缺失。
- SAM2 stronger checkpoint 可用或明确记录缺失。
- 20-case sanity benchmark 完成。

### Milestone B：医学难度分层

目标：

```text
把 IDRID 和 MonuSeg 按目标大小、对比度、实例密度分层。
```

完成标准：

- 每个 case 有 difficulty metadata。
- leaderboard 可以按 difficulty bucket 展开。

### Milestone C：Prompt 协议复现

目标：

```text
实现论文式 prompt protocol，而不是手写单个点/框。
```

完成标准：

- box_tight / box_loose / box_shift。
- click_1p / click_3p / click_1p1n。
- text template set。
- reference support set。

### Milestone D：模型主矩阵

目标：

```text
跑出 SAM, MedSAM, SAM2, SAM3, Matcher 的可解释对比。
```

完成标准：

- 每个 unsupported 组合有明确记录。
- 每个 failed 组合有错误日志。
- 每个 ok 组合有 mask/overlay/metadata。

### Milestone E：失败诊断

目标：

```text
把低分样本变成 Phase 3 的创新证据。
```

完成标准：

- failure tags。
- worst-case gallery。
- 每个 Phase 3 候选方向都有对应失败案例。

## 7. Phase 3 方向如何由 Phase 2 决定

### 如果强权重仍然输给 MedSAM

Phase 3 应优先：

```text
医学 adapter / LoRA / prompt-aware fine-tuning
```

### 如果 SAM3 text 语义不稳定

Phase 3 应优先：

```text
结构化医学 text prompt + 医学词表 + mask reranking
```

### 如果 SAM2 在序列中漂移

Phase 3 应优先：

```text
跨帧/跨切片 memory selection
```

### 如果 click/box 对小目标不稳定

Phase 3 应优先：

```text
自动 prompt generation + iterative correction
```

### 如果 Matcher 对 support 极敏感

Phase 3 应优先：

```text
reference retrieval + support quality scoring
```

## 8. 重构后的 Phase 2 一句话版本

```text
第 2 阶段不是简单比较 SAM、MedSAM、SAM2、Matcher、SAM3 谁分数高，
而是建立一个医学 promptable segmentation 诊断基准，
系统评估强权重、prompt 协议、医学难度、文本语义、参考样本和时序记忆对分割结果的影响，
最终用失败模式反推 Phase 3 的创新模块。
```

## 9. 当前最推荐的下一步

先做最小但论文味最强的实验：

```text
IDRID 20 cases + MonuSeg 20 cases
SAM-B / SAM-H / MedSAM / SAM2-Tiny / SAM2-strong
box_tight / click_1p / click_3p
按 target size 和 contrast 分层
```

这个实验能直接回答：

```text
是不是只是我们当前下载的权重不够强？
```

如果强权重仍然不能解决小目标、低对比、多实例问题，再进入 SAM3 text、Matcher reference 和 SAM2 memory 的细分实验。
