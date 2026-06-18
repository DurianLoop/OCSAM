# paper3：面向小项目发表的 SAM 医学分割论文阅读笔记

## 0. 选文标准

这一批论文和 `papers2` 的定位不同。

`papers2` 偏 foundation model / 大 benchmark / 顶会顶刊主线，适合理解领域天花板，但不适合我们这种希望用较少时间做出可发表创新的小项目直接照搬。

`papers3` 的选文标准是：

- 更接近 BIBM / MICCAI / MICCAI workshop / CVPR workshop / 医学图像小而清楚的论文风格。
- 重点不是做最大模型，而是找最小创新单元。
- 优先选择 prompt 自动化、prompt 鲁棒性、轻量 adapter、少量数据微调、视频/手术场景鲁棒性、伪标签等可快速落地的方向。
- 不强行声称每篇都是 CCF-B 正式主会论文；这里的目标是学习“BIBM/MICCAI 级别小论文怎么设计实验和创新”。

## 1. 下载清单

| 序号 | 文件 | 主题 | 年份 + 会议/期刊/等级标签 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | `01_SAMUS_AutoSAMUS_Ultrasound_Auto_Prompt_MICCAI2024.pdf` | 超声 SAM 适配 + 自动 prompt | 2024 arXiv 预印本；MICCAI/BIBM 风格小论文定位 | https://arxiv.org/abs/2309.06824 |
| 2 | `02_PP_SAM_Perturbed_Prompts_Polyp_CVPRW2024.pdf` | 边界框 prompt 扰动鲁棒训练 | 2024 arXiv 预印本；CVPR Workshop/BIBM 风格鲁棒性小论文定位 | https://arxiv.org/abs/2405.16740 |
| 3 | `03_Surgical_DeSAM_Auto_Box_Robotic_Surgery.pdf` | 检测器自动生成 box prompt + SAM | 2024 arXiv 预印本；MICCAI EndoVis/手术影像方向小论文定位 | https://arxiv.org/abs/2404.14040 |
| 4 | `04_SAM2_Robotic_Surgery_Robustness_Generalization.pdf` | SAM2 手术视频鲁棒性评估 | 2024 arXiv 预印本；MICCAI EndoVis/手术视频 benchmark 定位 | https://arxiv.org/abs/2408.04593 |
| 5 | `05_SAM2_Surgical_Video_Nonadversarial_Robustness.pdf` | SAM2 非对抗真实退化鲁棒性 | 2024 arXiv 预印本；MICCAI EndoVis 2024 SegSTRONG-C 相关评测定位 | https://arxiv.org/abs/2408.04098 |
| 6 | `06_SAM_Meets_Robotic_Surgery_LoRA_SurgicalSAM.pdf` | SAM 手术场景评测 + LoRA SurgicalSAM | 2023 arXiv 预印本；MICCAI/EndoVis 手术影像评测 + 轻量适配定位 | https://arxiv.org/abs/2308.07156 |
| 7 | `07_SamDSK_Domain_Knowledge_Semi_Supervised.pdf` | SAM + 医学先验 + 半监督伪标签 | 2023 arXiv 预印本；BIBM/MICCAI 半监督医学分割风格定位 | https://arxiv.org/abs/2308.13759 |
| 8 | `08_Uncertainty_Aware_SAM_Adapter_Ambiguous_Medical.pdf` | 不确定性感知 adapter / 多专家标注 | 2025 IEEE TMI 稿件/预印本倾向；期刊级医学分割定位 | https://arxiv.org/abs/2403.10931 |
| 9 | `09_SAM_Melanoma_WSI_Dynamic_Prompting.pdf` | WSI 黑色素瘤动态 prompt 策略 | 2024 arXiv 预印本；MICCAI/BIBM 病理 WSI 自动 prompt 风格定位 | https://arxiv.org/abs/2410.02207 |
| 10 | `10_Polyp_SAM_Transfer_SAM_for_Polyp.pdf` | 息肉分割的 SAM 迁移微调 | 2023 arXiv 预印本；BIBM/MICCAI 入门级任务迁移 baseline 定位 | https://arxiv.org/abs/2305.00293 |

文本抽取文件在：

```text
D:\SAM\docs\papers3\text
```

## 2. 逐篇中文笔记

### 1. SAMUS / AutoSAMUS：超声图像的 SAM 自动 prompt

文件：

```text
01_SAMUS_AutoSAMUS_Ultrasound_Auto_Prompt_MICCAI2024.pdf
```

标签：

```text
2024 arXiv 预印本；MICCAI/BIBM 风格小论文定位；可借鉴为自动 prompt + 轻量适配方向。
```

核心问题：

SAM 在超声图像上容易退化，原因包括低对比、边界弱、小目标、局部纹理重要。同时，临床场景不希望每张图都人工点或框。

方法要点：

- 在 SAM 图像编码器旁边加一个 CNN 分支，用来补足 ViT 对局部纹理和边界的不足。
- 使用 feature adapter 和 position adapter，把自然图像 SAM 迁移到超声域。
- 进一步提出 AutoSAMUS，用 auto prompt generator 替代人工 prompt。

实验风格：

- 任务集中在超声图像。
- 数据量比我们大，但思路很朴素：不是重新造大模型，而是“局部纹理补偿 + 自动 prompt”。

可借鉴点：

```text
医学图像中，prompt 自动化本身就是创新点。
```

我们可落地的低配版本：

- 不训练完整 AutoSAMUS。
- 用已有分割/阈值/显著性/连通域生成候选点和候选框。
- 在 SAM/MedSAM/SAM2 上比较：
  - 人工 box。
  - GT box。
  - 自动 box。
  - 自动 point/grid prompt。

适合我们的创新方向：

```text
结构化医学 prompt 自动生成。
```

风险：

- 如果完全不训练，自动 prompt 质量可能一般。
- 可以把目标降低为“自动 prompt 质量诊断 + 简单生成策略”，这更适合小项目。

### 2. PP-SAM：用扰动 prompt 做鲁棒适配

文件：

```text
02_PP_SAM_Perturbed_Prompts_Polyp_CVPRW2024.pdf
```

标签：

```text
2024 arXiv 预印本；CVPR Workshop/BIBM 风格鲁棒性小论文定位；可借鉴为 prompt 扰动鲁棒性方向。
```

核心问题：

医生给的 bounding box 不可能总是贴合目标。SAM/MedSAM 如果只在精准 box 下评价，论文结论会过于理想。

方法要点：

- 用少量息肉图像 fine-tune SAM。
- 训练时对 box prompt 做随机扰动。
- 目标是让模型对不准的 box 更鲁棒。
- 做 1-shot、5-shot、10-shot 等低数据量实验。

实验风格：

- 非常适合小项目。
- 创新点很小，但很清楚：prompt perturbation。
- 评估也直接：不同扰动强度下 Dice 是否下降更慢。

可借鉴点：

```text
不要只追求最高 Dice；可以把 prompt 鲁棒性作为核心 claim。
```

我们可落地方案：

- 在 IDRID 和 MonuSeg 上生成：
  - tight box。
  - loose box。
  - shifted box。
  - partial box。
- 对 SAM/MedSAM/SAM2/SAM3 比较 prompt sensitivity。
- 进一步训练一个轻量 adapter 或只做后处理，让模型对 box 扰动更稳。

最适合我们的论文 claim：

```text
医学图像 promptable segmentation 对提示误差非常敏感；通过扰动感知 prompt 协议或轻量适配，可以显著提高交互鲁棒性。
```

投入产出比：

非常高。这个方向实验量可控、创新容易讲清楚。

### 3. Surgical-DeSAM：检测器自动生成 box prompt

文件：

```text
03_Surgical_DeSAM_Auto_Box_Robotic_Surgery.pdf
```

标签：

```text
2024 arXiv 预印本；MICCAI EndoVis/手术影像方向小论文定位；可借鉴为自动 box prompt 方向。
```

核心问题：

手术视频中逐帧人工 prompt 不现实。对于实时部署，需要自动产生 prompt。

方法要点：

- 用 DETR/Swin-DETR 检测器预测器械 box。
- 把自动 box 送入 SAM 的 prompt encoder / mask decoder。
- 通过 decoupling 方式让检测器和 SAM mask 预测配合。

实验风格：

- 使用 EndoVis 2017/2018。
- 很典型的小论文路线：检测器 + SAM，解决人工 prompt 不现实的问题。

可借鉴点：

```text
自动 box prompt 是很实用的医学 SAM 创新。
```

我们可落地低配版本：

- 不训练 DETR。
- 用粗分割模型、传统图像处理、SAM 自动 mask、或者轻量 detector 产生候选 box。
- 对候选 box 进行 quality scoring，再调用 SAM/MedSAM。

在我们项目中的对应：

- IDRID：先找候选病灶区域，再生成 box。
- MonuSeg：先找 nuclei blob，再生成多实例框或点。

可写成：

```text
AutoBox-SAM for medical promptable segmentation。
```

风险：

- 如果候选检测器太弱，结果会拖累。
- 可以先把自动 box 作为 Phase 2 诊断任务，再进入 Phase 3 小创新。

### 4. SAM2 in Robotic Surgery：把 SAM2 当强 baseline 做鲁棒性诊断

文件：

```text
04_SAM2_Robotic_Surgery_Robustness_Generalization.pdf
```

标签：

```text
2024 arXiv 预印本；MICCAI EndoVis/手术视频 benchmark 定位；可借鉴为 SAM2 医学视频鲁棒性诊断。
```

核心问题：

SAM2 有视频 memory，但在手术视频中是否真的稳？不同 prompt 和真实退化下表现如何？

方法要点：

- 在 EndoVis 2017/2018 上测试 SAM2。
- 比较 1-point prompt 和 box prompt。
- 比较静态图像和视频序列。
- 评估 corruption / perturbation 下的鲁棒性。

实验风格：

- 创新不在模型结构，而在系统实验。
- 这很适合小项目：把 SAM2 放到具体医学场景，找失败规律。

可借鉴点：

```text
一个强 benchmark + failure analysis 本身可以形成 BIBM/MICCAI workshop 级别论文。
```

我们可落地方案：

- 把 3D 医学切片或视频看成 sequence。
- 设计：
  - first-frame click。
  - first-frame box。
  - sparse-frame correction。
  - worst-frame correction。
- 记录 mask drift、area jitter、disappear count。

适合我们的 Phase 2 调整：

```text
把 SAM2 的评价重点从“能不能出 GIF”升级为“memory 何时帮忙，何时漂移”。
```

### 5. SAM2 surgical video robustness：非对抗真实退化

文件：

```text
05_SAM2_Surgical_Video_Nonadversarial_Robustness.pdf
```

标签：

```text
2024 arXiv 预印本；MICCAI EndoVis 2024 SegSTRONG-C 相关评测定位；可借鉴为非对抗退化鲁棒性实验。
```

核心问题：

医学视频中会有烟雾、出血、低亮度、背景变化。模型要面对的是 non-adversarial corruption，而不是传统干净测试集。

方法要点：

- 使用 MICCAI EndoVIS 2024 SegSTRONG-C 类数据。
- 比较正常视频和退化视频。
- 比较 frame-wise prompt 和 frame-sparse prompt。
- 发现 frame-sparse prompting 反而可能更好，因为 SAM2 可以利用时间建模。

可借鉴点：

```text
不要只做干净数据 benchmark；退化鲁棒性是很好的小论文切入点。
```

我们可落地版本：

- 对 IDRID/MonuSeg 图像做人为退化：
  - blur。
  - low contrast。
  - noise。
  - brightness shift。
  - stain/color shift。
- 对视频/切片序列做：
  - frame dropout。
  - contrast drift。
  - local occlusion。
- 比较 SAM、MedSAM、SAM2、SAM3 在退化下的性能下降。

高性价比创新：

```text
医学 SAM 的 non-adversarial robustness benchmark + robust prompt strategy。
```

### 6. SAM Meets Robotic Surgery / SurgicalSAM：LoRA 小适配

文件：

```text
06_SAM_Meets_Robotic_Surgery_LoRA_SurgicalSAM.pdf
```

标签：

```text
2023 arXiv 预印本；MICCAI/EndoVis 手术影像评测 + LoRA 轻量适配定位；可借鉴为“评测 + 小修复”论文结构。
```

核心问题：

原始 SAM 在手术场景中 box prompt 还可以，但 point prompt 和无 prompt 不稳定；遇到血液、反光、模糊等退化会明显下降。

方法要点：

- 系统评估 SAM 在手术数据中的泛化和鲁棒性。
- 测 point、box、unprompted。
- 用 LoRA fine-tune，提出 SurgicalSAM，使其可以更适应手术域。

实验风格：

- 先做诊断，再做小适配。
- 非常适合我们：先证明 raw SAM 的失败，再给一个小修补。

可借鉴点：

```text
“评测 + 轻量 LoRA 修复”是很稳的小论文结构。
```

我们可落地方案：

- Phase 2：证明 SAM/MedSAM/SAM2 在低对比、小目标、多实例上有系统性失败。
- Phase 3：只训练轻量 LoRA / adapter 或 prompt reranker，不训练全模型。

推荐 claim：

```text
通过针对医学失败模式的轻量适配，提高 SAM 在小目标和低对比结构上的稳定性。
```

### 7. SamDSK：SAM + 医学先验做半监督伪标签

文件：

```text
07_SamDSK_Domain_Knowledge_Semi_Supervised.pdf
```

标签：

```text
2023 arXiv 预印本；BIBM/MICCAI 半监督医学分割风格定位；可借鉴为 SAM proposal + 医学先验伪标签筛选。
```

核心问题：

医学标注少，但未标注图像多。SAM 可以生成 proposal，但不能盲信，需要医学先验筛选。

方法要点：

- 初始少量标注训练一个 segmentation model。
- 用该模型和 SAM 在未标注图像上生成候选。
- 用 domain-specific knowledge 筛选/组合 SAM proposals。
- 迭代扩充 labeled set。

实验风格：

- 任务包括超声乳腺、息肉、皮肤病灶。
- 创新点是“SAM proposal + 医学先验 + 伪标签筛选”。

可借鉴点：

```text
SAM 不一定直接作为最终模型，也可以作为伪标签生成器。
```

我们可落地低配版本：

- 用 SAM automatic masks 生成候选 mask。
- 用医学先验打分：
  - 面积范围。
  - 连通域数量。
  - 形状紧致度。
  - 与粗模型概率图的一致性。
- 选择高置信 mask 作为 pseudo-label。

适合我们的小创新：

```text
医学先验约束的 SAM pseudo-label selection。
```

优点：

- 不依赖 SAM3/SAM2 权重是否最强。
- 对小数据项目很友好。

### 8. Uncertainty-aware SAM Adapter：多专家不确定性

文件：

```text
08_Uncertainty_Aware_SAM_Adapter_Ambiguous_Medical.pdf
```

标签：

```text
2025 IEEE TMI 稿件/预印本倾向；期刊级医学分割定位；对我们来说可低配化为 prompt-induced uncertainty。
```

核心问题：

医学边界经常模糊，不同专家可能给出不同但合理的标注。普通 SAM adaptation 输出一个确定 mask，忽略了多专家不确定性。

方法要点：

- 用 uncertainty-aware adapter。
- 从多专家标注分布中学习多个 plausible masks。
- 用条件变分/随机采样机制表达不确定性。

实验风格：

- 这篇比其他 paper3 文章更大、更偏 TMI。
- 但它给我们一个可以低配复现的思路。

我们可借鉴的低配版本：

```text
不用训练复杂不确定性模型，而是用多 prompt 采样生成 uncertainty map。
```

具体做法：

- 对同一个目标生成多组 prompt：
  - box jitter。
  - point jitter。
  - negative point variation。
- 得到多张 mask。
- 计算像素级 mask variance / entropy。
- 用 uncertainty map 识别边界不确定区域。

可写创新：

```text
Prompt-induced uncertainty for medical SAM segmentation。
```

适合我们吗：

很适合。因为我们已经要做 prompt perturbation benchmark，顺手就能产生不确定性图。

### 9. Melanoma WSI dynamic prompting：从粗分割生成动态 prompt

文件：

```text
09_SAM_Melanoma_WSI_Dynamic_Prompting.pdf
```

标签：

```text
2024 arXiv 预印本；MICCAI/BIBM 病理 WSI 自动 prompt 风格定位；可借鉴为连通域几何驱动动态 prompt。
```

核心问题：

WSI 太大，不能直接把整张图丢进 SAM；而且单一点 prompt 不足以覆盖大面积、不规则病灶。

方法要点：

- 先用 SegFormer 得到粗 mask。
- 从粗 mask 的连通域生成 prompt。
- 根据连通域形状动态选择 centroid prompt 或 grid prompt。
- 过滤低置信区域，避免错误 prompt。
- 再用 EfficientSAM/SAM 细化。

实验风格：

- 非常符合小项目：粗模型 + prompt 生成 + SAM refinement。
- 创新不复杂，但很实用。

可借鉴点：

```text
根据目标几何形状动态选择 prompt 类型。
```

我们可落地方案：

- 对 MonuSeg：
  - 小连通域：centroid point。
  - 大/长条/粘连区域：grid points 或 box。
- 对 IDRID：
  - 小病灶：centroid point。
  - 大片渗出：box + grid points。
- 输出 prompt strategy ablation。

推荐 claim：

```text
结构感知医学 prompt 策略可以降低人工交互并提升 SAM 系列模型稳定性。
```

### 10. Polyp-SAM：最直接的 SAM 迁移微调

文件：

```text
10_Polyp_SAM_Transfer_SAM_for_Polyp.pdf
```

标签：

```text
2023 arXiv 预印本；BIBM/MICCAI 入门级任务迁移 baseline 定位；可借鉴为单病种 SAM fine-tuning baseline。
```

核心问题：

原始 SAM 可以被迁移到单一医学任务上，问题是微调哪些模块、跨数据集泛化如何。

方法要点：

- 在息肉数据上 fine-tune SAM。
- 比较只 fine-tune mask decoder 和 fine-tune encoders + decoder。
- 用多个公开息肉数据集测试跨数据集泛化。
- 使用 Dice 和 mIoU。

实验风格：

- 很像入门级 BIBM/MICCAI 小论文：目标单一、实验清楚、创新不大。

可借鉴点：

```text
单病种/单模态的 SAM 迁移仍然可以作为 baseline 或小论文起点。
```

我们可落地方案：

- 如果想最快出结果，可以选 MonuSeg 或 IDRID 做一个任务特化 SAM。
- 但风险是创新偏弱，容易被认为只是“又 fine-tune 一次 SAM”。

更好的改法：

```text
不要只做 fine-tune；叠加 prompt 鲁棒性、自动 prompt 或 uncertainty。
```

## 3. 对我们项目最有用的 5 个套路

### 套路 A：Prompt 扰动鲁棒性

参考：

- PP-SAM。
- SAM2 surgery robustness。
- Uncertainty-aware adapter 的低配思路。

最小实现：

```text
对 box / point 做扰动 -> 多次推理 -> 统计 Dice 均值、方差、最差值、mask variance。
```

可发表点：

```text
医学 SAM 的 prompt sensitivity diagnosis + robust prompt strategy。
```

### 套路 B：自动 prompt 生成

参考：

- AutoSAMUS。
- Surgical-DeSAM。
- Melanoma dynamic prompting。

最小实现：

```text
粗 mask / 连通域 / 显著性 -> centroid / grid / box prompt -> SAM/MedSAM/SAM2 refinement。
```

可发表点：

```text
结构感知自动 prompt 生成，减少人工交互成本。
```

### 套路 C：轻量 LoRA / Adapter

参考：

- SurgicalSAM。
- SAMUS。
- Uncertainty-aware adapter。

最小实现：

```text
冻结 SAM 大部分参数，只训练小 adapter 或 LoRA。
```

可发表点：

```text
针对医学失败模式的小参数适配。
```

### 套路 D：SAM2 稀疏提示和 memory 分析

参考：

- SAM2 surgical robustness。
- SAM2 non-adversarial robustness。

最小实现：

```text
first-frame prompt vs frame-sparse prompt vs frame-wise prompt。
```

可发表点：

```text
医学序列中，少量高质量 correction prompt 比逐帧提示更有效。
```

### 套路 E：SAM 伪标签筛选

参考：

- SamDSK。

最小实现：

```text
SAM automatic masks + 医学先验规则 -> 高置信 pseudo-label -> 训练小模型或 adapter。
```

可发表点：

```text
用医学先验筛选 SAM proposal，降低标注成本。
```

## 4. 最推荐我们走的方向

综合投入产出比，我建议不要先做大规模模型训练，而是选一个组合创新：

```text
结构感知自动 prompt + prompt 扰动不确定性 + SAM2/MedSAM refinement
```

中文描述：

```text
我们首先从医学图像的粗候选区域中自动生成结构化提示，包括中心点、网格点和边界框；
然后通过提示扰动得到多次 SAM 系列模型输出，构建提示诱导不确定性图；
最后根据不确定性选择需要修正的区域或选择更稳定的 mask。
```

这条路线的优点：

- 不需要训练很大的模型。
- 可以直接用现有网页和 Phase 1 工作台。
- 论文 claim 不会太大，但足够清楚。
- 能自然连接 SAM、MedSAM、SAM2、SAM3、Matcher。
- 可以在 IDRID 和 MonuSeg 上快速验证。

建议英文题目方向：

```text
Structure-aware Prompt Generation and Prompt-induced Uncertainty for Medical Segment Anything Models
```

或者更 BIBM 风格：

```text
Robust Automatic Prompting for Segment Anything Models in Medical Image Segmentation
```

## 5. 建议实验设计

### 数据集

先用我们已有数据：

```text
IDRID    眼底病灶，小目标、低对比
MonuSeg  细胞核，多实例、密集、边界接触
```

### Baselines

```text
SAM ViT-B
SAM ViT-H
MedSAM
SAM2 Tiny / SAM2 strong
Matcher
```

可选：

```text
SAM3 text
```

### Prompt 协议

```text
GT box
perturbed box
manual-like center point
auto centroid point
auto grid points
auto box from connected components
```

### 我们方法

```text
结构感知自动 prompt：
  小目标 -> centroid point
  大目标 -> box
  长条/不规则目标 -> grid points
  多实例密集目标 -> connected-component prompts

提示不确定性：
  对 prompt 做 jitter
  运行多次 SAM/MedSAM/SAM2
  计算 mask variance
  选择稳定 mask 或标记需人工修正区域
```

### 指标

```text
Dice
IoU
Boundary F1
Prompt count
Runtime
Worst-case Dice
Prompt sensitivity
Uncertainty-error correlation
```

### Ablation

```text
无自动 prompt
只 centroid
centroid + grid
centroid + grid + box
不使用 prompt uncertainty
使用 prompt uncertainty reranking
```

## 6. 最快成文结构

### Introduction

讲清楚：

- SAM 系列很强，但医学图像 prompt 质量极大影响结果。
- 医学场景中人工 prompt 成本高、误差不可避免。
- 我们做结构感知自动 prompt，并用 prompt 扰动估计不确定性。

### Method

三个模块：

1. Medical candidate extraction。
2. Structure-aware prompt generation。
3. Prompt-induced uncertainty and mask selection。

### Experiments

四组：

1. 与 SAM/MedSAM/SAM2 的比较。
2. 自动 prompt vs GT prompt vs perturb prompt。
3. 不确定性是否能预测失败。
4. 消融实验。

### Claim

不要写：

```text
We propose a better SAM.
```

要写：

```text
We improve the reliability and interaction efficiency of SAM-style medical segmentation by structure-aware automatic prompting and prompt-induced uncertainty estimation.
```

## 7. 和 Phase 2 的关系

`papers3` 给 Phase 2 的调整建议：

1. Phase 2 不只做 leaderboard。
2. 必须记录 prompt sensitivity。
3. 必须比较自动 prompt 和人工/GT prompt。
4. 必须加入退化或扰动鲁棒性。
5. 先做小而稳的 IDRID + MonuSeg，不要一上来做 20 个数据集。

建议把 Phase 2 目标进一步收敛为：

```text
在 IDRID 和 MonuSeg 上建立 prompt sensitivity benchmark，
比较 SAM/MedSAM/SAM2/Matcher/SAM3 在不同 prompt 质量下的表现，
并验证结构感知自动 prompt 和 prompt-induced uncertainty 是否能提高稳定性和交互效率。
```

这个目标比 `papers2` 那版更小、更适合快速发论文。
