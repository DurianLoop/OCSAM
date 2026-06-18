# 第 2 阶段：Benchmark 实验方案

> 更新说明：在阅读 `docs/papers2` 论文包后，Phase 2 已重构为论文驱动版本。后续优先参考 `docs/phase2_reconstructed_from_papers.md`；本文保留作为第一版工程化 benchmark 草案。

## 1. 阶段目标

第 2 阶段的目标不是继续堆模型，而是把第 1 阶段已经能跑的模型放进同一个可复现实验协议里，回答三个问题：

1. 同一张医学图像、同一种 prompt 下，哪个模型更准？
2. 哪些场景会让模型明显失败？
3. 后续第 3 阶段应该优先改 prompt、记忆机制、adapter，还是模型本体？

核心输出应该是一套稳定的 benchmark：

```text
dataset + case_id + model + prompt_protocol -> masks + metrics + failure_tags + artifacts
```

最终要得到可用于论文和工程决策的表格、图像对比、失败案例库和复现实验脚本。

## 2. Benchmark 原则

### 2.1 公平性

所有模型必须尽量使用同一份输入图像、同一份 ground truth、同一套 prompt 生成规则。

对于不能支持某类 prompt 的模型，不强行伪装成支持，而是标记为 `unsupported`。例如：

- SAM2 当前不做原生 text prompt。
- MedSAM 以 box prompt 为主。
- Matcher 使用 reference image + reference mask 或 reference box，不属于普通 click/box prompt。
- SAM3 的核心优势是 text/open-vocabulary prompt，不应只拿 box prompt 评价它。

### 2.2 可复现性

每一次评测必须记录：

- 代码版本。
- 权重路径和权重 hash。
- 数据集路径和样本列表。
- prompt 生成 seed。
- 模型参数。
- 推理时间。
- GPU/CPU 环境。
- 后处理参数。

所有输出都落盘，不能只显示在网页里。

### 2.3 医学任务优先

通用数据集可以用于 sanity check，但核心结论必须来自医学图像。

当前建议主线：

- IDRID：眼底病灶分割。
- MonuSeg：病理细胞核分割。
- 后续扩展：CT/MRI 2.5D 或 3D 切片任务。

COCO、FSS-1000 可以帮助验证 Matcher 的 one-shot 行为，但不能作为医学 SAM 改进的主证据。

## 3. 模型范围

### 3.1 必测模型

| 模型 | 当前角色 | 主要 prompt | 评价重点 |
| --- | --- | --- | --- |
| SAM ViT-B | 通用 promptable segmentation baseline | click, box, heuristic text | 通用 SAM 在医学图像上的基础能力 |
| MedSAM ViT-B | 医学 box-prompt baseline | box | 医学图像上 box prompt 的强基线 |
| SAM2.1 Tiny | 图像 + 视频 prompt baseline | click, box, video first-frame prompt | 时序传播和视频交互能力 |
| Matcher | one-shot segmentation baseline | reference image + mask/box | 少样本迁移和外观匹配能力 |
| SAM3 | text/open-vocabulary baseline | text, optional click/box | 文本提示和开放词汇能力 |

### 3.2 候选扩展模型

这些模型不阻塞 Phase 2，但应在表格中预留位置：

- MedSAM2
- Medical SAM2
- SAM-Med2D
- MedSAM3 或 SAM3 医学变体
- One-Prompt / MedSAM-Agent / OSAM-Fundus 等 prompt 自动化方法

扩展模型必须先适配统一接口：

```python
predict(image_or_video, prompt, options) -> Prediction
```

其中 `Prediction` 至少包含：

```python
mask: np.ndarray
masks: list[np.ndarray] | None
boxes: list[list[float]] | None
scores: list[float] | None
runtime_ms: float
metadata: dict
```

## 4. 数据集与任务定义

### 4.1 IDRID

路径：

```text
D:\SAM\code\datasets\IDRID
```

任务：

```text
眼底糖尿病视网膜病变病灶分割
```

建议标签协议：

- `lesion_all`：合并所有非背景病灶。
- `red_lesion`：如果能从 mask 颜色稳定恢复，单独评价红色病灶。
- `bright_lesion`：如果能从 mask 颜色稳定恢复，单独评价亮性病灶。

主要困难：

- 病灶小。
- 前景极稀疏。
- 边界低对比。
- 假阳性区域多。

优先指标：

- Dice
- IoU
- lesion-level recall
- false positive area ratio

### 4.2 MonuSeg

路径：

```text
D:\SAM\code\datasets\MonuSeg
```

任务：

```text
组织病理细胞核分割
```

建议标签协议：

- `nuclei_all`：所有细胞核合并为前景。
- `nuclei_instance_optional`：如果 instance id 可以保留，则补充实例级指标。

主要困难：

- 高密度小目标。
- 相邻细胞核贴连。
- 染色差异。
- 单点 prompt 容易只分出局部。

优先指标：

- Dice
- IoU
- Boundary F-score
- instance split/merge error

### 4.3 视频任务

当前 Phase 2 视频任务先用于 SAM2 行为检查，不作为所有模型公平对比的主表。

建议任务：

- 对 demo 视频抽取首帧 prompt。
- 传播到固定帧数，例如 8、16、32、64。
- 如果没有逐帧 GT，先做人眼质检和稳定性指标。

视频指标分两级：

1. 有 GT 时：
   - frame-wise Dice
   - frame-wise IoU
   - temporal consistency
2. 无 GT 时：
   - mask area jitter
   - centroid drift
   - prompt-to-mask visual pass/fail
   - failure tag

## 5. Prompt 协议

### 5.1 Box Prompt

Box 从 ground truth mask 自动生成。

基础协议：

```text
box = bounding_box(gt_mask)
```

扰动协议：

```text
box_tight      = gt bbox
box_loose_10   = bbox 向外扩张 10%
box_loose_25   = bbox 向外扩张 25%
box_shift_10   = bbox 随机平移 10%
box_partial    = bbox 裁掉一部分目标
```

目的：

- 测模型对人工框误差的鲁棒性。
- 区分“只要框准就很强”和“框略差就崩”的模型。

适用模型：

- SAM
- MedSAM
- SAM2 image
- SAM3 optional box

### 5.2 Click Prompt

Click 从 GT 自动生成。

单点协议：

```text
positive_1 = distance_transform(gt_mask) 的最大点
```

多点协议：

```text
positive_k = 在前景中按距离变换或 farthest point sampling 选点
negative_k = 在 gt 边界外或模型误分区域附近选点
```

建议设置：

| 协议 | 正点 | 负点 | 用途 |
| --- | --- | --- | --- |
| click_1p | 1 | 0 | 最低交互成本 |
| click_3p | 3 | 0 | 多目标或复杂前景 |
| click_1p1n | 1 | 1 | 排除邻近误分 |
| click_3p3n | 3 | 3 | 强交互上限 |

适用模型：

- SAM
- SAM2 image
- SAM2 video first frame
- SAM3 optional click

### 5.3 Text Prompt

Text prompt 必须结构化，不要只写随意自然语言。

建议字段：

```text
target_name
organ_or_modality
appearance
location_optional
negative_hint_optional
```

IDRID prompt 示例：

```text
lesion
retinal lesion
small red retinal lesion
bright retinal exudate
diabetic retinopathy lesion
```

MonuSeg prompt 示例：

```text
nucleus
cell nucleus
histopathology nuclei
purple stained cell nuclei
clustered nuclei
```

适用模型：

- SAM3：主要评价对象。
- SAM：当前工作台中的 heuristic text 只能作为弱 baseline，必须在表格中标注。
- SAM2：当前不评价 text。
- MedSAM：当前不评价 text。

### 5.4 Matcher Reference Prompt

Matcher 的 prompt 不等同于 click/box，而是 in-context example。

输入：

```text
support_image
support_mask 或 support_box
target_image
matcher_version
```

参考样本选择协议：

1. Same-dataset nearest：同一数据集内选择外观相近的 support。
2. Same-dataset random：同一数据集内随机 support。
3. Cross-dataset stress：跨数据集 support，用于压力测试。
4. Oracle support：人工挑选高质量 support，用作上限。

Matcher 三个 version 的含义：

| Version | 任务偏好 | 适用情况 | 风险 |
| --- | --- | --- | --- |
| v1 | multiple instances | 目标类别在 target 中出现多个实例，例如多个细胞核 | 容易合并过多相似区域 |
| v2 | whole object | 希望分出单个完整目标 | 多实例场景可能召回不足 |
| v3 | object part | support mask 是对象局部，或只想迁移局部结构 | 容易只分出局部而非完整对象 |

在医学任务里：

- MonuSeg 多细胞核可优先看 v1。
- 单个病灶或单个结构可优先看 v2。
- 局部病灶、局部组织结构可试 v3。

## 6. 指标体系

### 6.1 像素级指标

必算：

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

建议补充：

```text
Specificity
False Positive Area Ratio
False Negative Area Ratio
```

### 6.2 边界指标

对于医学结构边界，建议补充：

```text
Boundary F-score
Hausdorff Distance 95% (HD95)
Average Symmetric Surface Distance (ASSD)
Normalized Surface Dice (NSD)
```

Phase 2 最低可先实现：

- Dice
- IoU
- Precision
- Recall
- Boundary F-score

HD95/ASSD/NSD 可作为第二轮增强。

### 6.3 交互效率指标

promptable segmentation 不只看准确率，还要看达到某个质量所需交互成本。

建议记录：

```text
number_of_clicks
box_used
text_used
reference_used
time_to_result_ms
model_init_ms
inference_ms
```

核心派生指标：

```text
Dice@1click
Dice@3click
Dice@box
Dice@text
Clicks to Dice >= 0.80
Clicks to Dice >= 0.90
```

### 6.4 稳定性指标

对同一图像进行 prompt 扰动，统计输出波动：

```text
mean_metric
std_metric
worst_metric
prompt_sensitivity = best_metric - worst_metric
```

对视频：

```text
area_jitter
centroid_jitter
mask_disappear_count
identity_switch_count
```

## 7. 失败模式标签

每个 bad case 应该记录失败原因，不能只记录分数。

建议标签：

| 标签 | 含义 |
| --- | --- |
| `miss_small_object` | 小目标漏检 |
| `over_segment_background` | 背景被过分割 |
| `under_segment_target` | 目标只分出一部分 |
| `merge_instances` | 多实例粘连 |
| `split_single_object` | 单个目标被拆开 |
| `boundary_leak` | 边界泄漏 |
| `prompt_sensitive` | prompt 轻微变化导致结果大变 |
| `text_mismatch` | 文本语义和输出不匹配 |
| `reference_mismatch` | Matcher support 与 target 匹配失败 |
| `temporal_drift` | 视频传播漂移 |
| `mask_disappear` | 视频中 mask 消失 |

失败标签来源：

1. 自动规则：低 Dice、高 FP、高 FN、mask 面积异常。
2. 人工复核：网页中点击标记。
3. 模型 metadata：记录 prompt、score、box、mask area。

## 8. 输出目录结构

建议所有 Phase 2 输出放在：

```text
D:\SAM\runs\phase2_benchmark
```

结构：

```text
runs/phase2_benchmark/
  configs/
    idrid_box.yaml
    idrid_click.yaml
    monuseg_box.yaml
    monuseg_click.yaml
    matcher_reference.yaml
    sam2_video.yaml
  manifests/
    idrid_cases.csv
    monuseg_cases.csv
    video_cases.csv
  predictions/
    {dataset}/{case_id}/{model}/{prompt_protocol}/mask.png
    {dataset}/{case_id}/{model}/{prompt_protocol}/overlay.png
    {dataset}/{case_id}/{model}/{prompt_protocol}/metadata.json
  metrics/
    pixel_metrics.csv
    boundary_metrics.csv
    interaction_metrics.csv
    video_metrics.csv
  reports/
    leaderboard.md
    failure_gallery.html
    per_dataset_summary.md
    ablation_prompt_protocol.md
```

## 9. 表格 Schema

### 9.1 case manifest

```csv
dataset,case_id,split,image_path,mask_path,modality,task,target_label,difficulty
```

示例：

```csv
IDRID,IDRiD_01,train,...jpg,...png,fundus,lesion_seg,lesion_all,small_sparse
MonuSeg,TCGA_xxx,test,...png,...png,pathology,nuclei_seg,nuclei_all,dense_instances
```

### 9.2 prediction metadata

```json
{
  "dataset": "MonuSeg",
  "case_id": "TCGA_xxx",
  "model": "sam",
  "prompt_protocol": "click_1p",
  "prompt": {
    "type": "click",
    "points": [[120, 88, 1]]
  },
  "runtime_ms": 184.2,
  "device": "cuda",
  "weights": "D:/SAM/assets/checkpoints/sam_vit_b.pth",
  "model_metadata": {}
}
```

### 9.3 metrics table

```csv
dataset,case_id,split,model,prompt_protocol,dice,iou,precision,recall,boundary_f1,runtime_ms,status,error,failure_tags
```

`status` 值：

```text
ok
unsupported
failed
timeout
invalid_gt
```

## 10. 实验矩阵

### 10.1 主实验：医学图像 box prompt

目的：

比较 SAM、MedSAM、SAM2、SAM3 在同一 box prompt 下的能力。

矩阵：

| Dataset | Prompt | Models |
| --- | --- | --- |
| IDRID | box_tight, box_loose_10, box_shift_10 | SAM, MedSAM, SAM2, SAM3 |
| MonuSeg | box_tight, box_loose_10, box_shift_10 | SAM, MedSAM, SAM2, SAM3 |

核心图表：

- 每个模型 Dice/IoU 平均值。
- box 扰动下的性能下降。
- 每个数据集 top failure cases。

### 10.2 主实验：医学图像 click prompt

目的：

衡量交互点数量对结果的影响。

矩阵：

| Dataset | Prompt | Models |
| --- | --- | --- |
| IDRID | click_1p, click_3p, click_1p1n, click_3p3n | SAM, SAM2, SAM3 |
| MonuSeg | click_1p, click_3p, click_1p1n, click_3p3n | SAM, SAM2, SAM3 |

核心图表：

- Dice vs clicks 曲线。
- 小目标/密集目标上的提升幅度。
- 正负点是否降低 FP。

### 10.3 主实验：text prompt

目的：

比较 SAM3 文本能力和 SAM heuristic text baseline。

矩阵：

| Dataset | Prompt | Models |
| --- | --- | --- |
| IDRID | lesion, retinal lesion, red lesion, bright lesion | SAM heuristic, SAM3 |
| MonuSeg | nucleus, cell nucleus, histopathology nuclei | SAM heuristic, SAM3 |

核心图表：

- 不同文本模板的性能差异。
- text mismatch 失败案例。
- 文本是否适合医学精细目标。

### 10.4 主实验：Matcher one-shot

目的：

评价 reference prompt 在医学图像中的少样本迁移能力。

矩阵：

| Dataset | Support | Versions |
| --- | --- | --- |
| IDRID | same-dataset random, oracle support | v1, v2, v3 |
| MonuSeg | same-dataset random, oracle support | v1, v2, v3 |

核心图表：

- support 质量对结果的影响。
- v1/v2/v3 在多实例、完整对象、局部结构上的差异。
- reference mismatch 失败案例。

### 10.5 视频实验：SAM2 propagation

目的：

评估 SAM2 在视频首帧 prompt 后的传播稳定性。

矩阵：

| Video type | Prompt | Model |
| --- | --- | --- |
| simple | click_1p, box | SAM2.1 Tiny |
| difficult | click_1p, click_3p, box | SAM2.1 Tiny |

核心图表：

- 每帧 mask 面积曲线。
- 中心点漂移曲线。
- 失败帧可视化。

## 11. 脚本设计

### 11.1 构建 manifest

建议脚本：

```text
code/benchmarks/build_manifest.py
```

功能：

```text
扫描数据集 -> 检查 image/mask 配对 -> 生成 CSV -> 统计尺寸和标签覆盖
```

命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\build_manifest.py `
  --dataset idrid `
  --root D:\SAM\code\datasets\IDRID `
  --out D:\SAM\runs\phase2_benchmark\manifests\idrid_cases.csv
```

### 11.2 生成 prompts

建议脚本：

```text
code/benchmarks/generate_prompts.py
```

功能：

```text
读取 manifest 和 GT mask -> 生成 box/click/text/reference prompt jsonl
```

输出：

```text
runs/phase2_benchmark/prompts/{dataset}_{protocol}.jsonl
```

### 11.3 批量推理

建议脚本：

```text
code/benchmarks/run_benchmark.py
```

命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_benchmark.py `
  --config D:\SAM\runs\phase2_benchmark\configs\monuseg_click.yaml
```

功能：

- 加载统一 adapters。
- 对 unsupported 组合写入 `status=unsupported`。
- 对失败样本写入 error，不中断整轮实验。
- 保存 mask、overlay、metadata。
- 支持 resume，避免重复跑已完成样本。

### 11.4 计算指标

建议脚本：

```text
code/benchmarks/evaluate_predictions.py
```

命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\evaluate_predictions.py `
  --pred-root D:\SAM\runs\phase2_benchmark\predictions `
  --manifest D:\SAM\runs\phase2_benchmark\manifests\monuseg_cases.csv `
  --out D:\SAM\runs\phase2_benchmark\metrics\pixel_metrics.csv
```

### 11.5 生成报告

建议脚本：

```text
code/benchmarks/make_report.py
```

输出：

```text
leaderboard.md
failure_gallery.html
per_dataset_summary.md
```

## 12. Web 集成

Phase 2 的网页不只是 demo，还应支持 benchmark 检查。

建议增加：

1. Benchmark tab。
2. Dataset selector。
3. Case selector。
4. Prompt protocol selector。
5. Model checklist。
6. Run selected。
7. Overlay comparison。
8. Metrics table。
9. Failure tag editor。
10. Export current case。

网页中的每一次运行都应该可以保存为与批量脚本同样的 artifact 结构，避免 demo 结果和 benchmark 结果割裂。

## 13. 质量控制

### 13.1 数据 QC

每个数据集先生成 QC 表：

```text
case_id
image_size
mask_size
mask_area_ratio
connected_components
has_empty_mask
has_invalid_mask
```

排除规则：

- 图像和 mask 尺寸不一致且无法修复。
- mask 完全为空。
- mask 颜色无法解析。
- case 文件损坏。

### 13.2 推理 QC

每个预测检查：

- mask 是否为空。
- mask 是否全图。
- mask 尺寸是否和原图一致。
- runtime 是否异常。
- 是否产生 NaN metric。

异常不删除，写入 `status` 和 `failure_tags`。

### 13.3 报告 QC

报告中必须包含：

- 成功样本数。
- 失败样本数。
- unsupported 组合数。
- 每个模型的平均 runtime。
- 每个数据集的 metric 均值和标准差。
- worst 20 cases。

## 14. 第一轮执行顺序

建议先跑最小闭环：

1. MonuSeg 10 个样本。
2. IDRID 10 个样本。
3. 只跑 box_tight。
4. 模型：SAM、MedSAM、SAM2。
5. 输出 Dice/IoU 和 overlay。

通过后扩展：

1. 加 click_1p 和 click_3p。
2. 加 SAM3 text。
3. 加 Matcher same-dataset random support。
4. 加 box 扰动。
5. 加完整数据集。
6. 加视频稳定性实验。

## 15. Phase 2 验收标准

Phase 2 完成时，应满足：

- 有统一 case manifest。
- 有自动 prompt 生成。
- 有批量推理脚本。
- 有指标计算脚本。
- 有至少 IDRID 和 MonuSeg 的结果表。
- 有 SAM、MedSAM、SAM2、Matcher、SAM3 的可比结果或明确 unsupported 记录。
- 有 failure gallery。
- 有视频传播稳定性初步报告。
- 所有结果可以从命令行复现。
- 网页可以加载某个 benchmark case 并复查模型输出。

## 16. 进入 Phase 3 的决策标准

Phase 2 结束后，按以下规则决定 Phase 3 创新重点：

如果 box prompt 下 MedSAM 明显强于 SAM：

```text
优先做医学 box/prompt 自动化，减少人工框成本。
```

如果 click prompt 下 SAM/SAM2 对小目标不稳定：

```text
优先做结构化点击策略和失败后自动补点。
```

如果 SAM2 视频传播漂移严重：

```text
优先做跨切片/跨帧 memory selection。
```

如果 SAM3 text 在医学目标上语义不稳定：

```text
优先做医学文本 prompt 模板、医学词表和候选 mask reranking。
```

如果 Matcher 对 support 选择极度敏感：

```text
优先做 reference retrieval 和 support quality scoring。
```

最终 Phase 3 的主线建议仍然是：

```text
医学 prompt 自动化 + SAM2 memory 选择 + 轻量 adapter
```

这条路线可以直接由 Phase 2 的失败模式支撑，而不是凭直觉提出新模型。
