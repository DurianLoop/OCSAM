# 医学 SAM / Matcher 入门地图

更新日期：2026-06-16

这个项目可以按两条线理解：

1. **通用分割基础模型线**：SAM -> SAM2 -> SAM3
2. **医学图像适配线**：MedSAM / One-Prompt / UM-SAM / OSAM-Fundus / MedSAM-Agent / MedSAM3

本仓库当前真正能直接运行的是 **Matcher + SAM/DINOv2/Semantic-SAM** 这一套；医学部分主要是把 IDRID、MonuSeg、DDR 等数据集接进来，做一示例/一提示/partial-to-full 风格实验。

## 一、SAM 家族主线

### SAM: Segment Anything

核心问题：给点、框、粗 mask 等 prompt，让模型输出目标 mask。

对应代码：

- `segment_anything/`
- `models/sam_vit_b.pth`
- `models/sam_vit_h_4b8939.pth`
- `demos/medical_sam_box_demo.py`

在本项目里的作用：作为 mask 生成器。Matcher 或医学 demo 先产生点/框/候选区域，再调用 SAM 解码出 mask。

### SAM2

核心变化：从静态图像扩展到图像和视频，增加视频/序列中的记忆传播能力。

本仓库状态：没有 SAM2 官方代码。`README.md` 里提到 Matcher 的 VOS 分支，但本地主要还是 SAM1 + Matcher。

如果后续要做医学 3D/视频，可把 SAM2 看作“跨切片/跨帧传播”的基础模型。

### SAM3

核心变化：从几何 prompt 进一步走向 **concept prompt**，也就是用短语、图像示例或二者组合来检测、分割、跟踪概念对象。

本仓库状态：没有 SAM3 代码或权重。SAM3 更像未来可替换的更强基座：把现在 Matcher 里的 `segment_anything/` 换成支持概念提示的 SAM3 系列模型。

## 二、医学 SAM / 医学分割适配线

### MedSAM: Segment Anything in Medical Images

核心问题：原始 SAM 在医学图像上经常不稳，因为医学图像和自然图像域差异大，器官/病灶边界也更细。MedSAM 用大规模医学图像-mask 数据对 SAM 做医学域适配。

和本仓库关系：

- 本仓库还没有 MedSAM 官方实现。
- 但 `demos/medical_sam_box_demo.py` 已经展示了“原始 SAM 直接做医学图像”的效果：MonuSeg 尚可，IDRID 微小病灶很差。这正是 MedSAM 这类工作的动机。

### One-Prompt to Segment All Medical Images

核心问题：不想每张图都人工点/框；希望只给一个带 prompt 的样例，就能推广到同任务新图像。

和本仓库关系：

- 思想上接近 Matcher：都强调用示例/上下文迁移到新图。
- Matcher 是通用视觉基础模型组合，不专门医学训练；One-Prompt 是医学任务范式和医学数据训练更强。

### Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching

核心问题：给一张 support 图和 support mask，不训练，分割 query 图中的同类对象。

对应代码：

- `matcher/`
- `dinov2/`
- `segment_anything/`
- `semantic_sam/`
- `main_oss.py`
- `gradio_demo/`

工作流：

1. support image + support mask 指明“我要什么”
2. DINOv2 提取 support/query 特征
3. Matcher 找 query 中相似 patch/点/框
4. SAM 或 Semantic-SAM 根据这些提示生成 mask

### Semantic-SAM

核心问题：提供更细粒度、多层级的分割能力，尤其适合 part segmentation。

对应代码：

- `semantic_sam/`
- `configs/semantic_sam_*.yaml`
- `models/swint_only_sam_many2many.pth`
- `matcher/Matcher_SemanticSAM.py`

和医学关系：如果目标是细碎结构、器官局部、病灶区域，它可能比普通 SAM 候选 mask 更细，但需要更重的依赖环境。

### UM-SAM

核心问题：用 SAM 作为教师或先验，通过知识蒸馏/无监督策略，把医学分割能力迁移给更适合医学场景的模型，减少人工标注依赖。

本仓库状态：没有明确的 UM-SAM 官方实现。它更像一个可以借鉴的训练方向：用当前 SAM/Matcher 输出的 pseudo mask 或候选区域训练医学学生模型。

### OSAM-Fundus

核心问题：眼底图像的 optic disc / optic cup 一示例或无训练分割。它和本地 IDRID/DDR 数据方向接近，但目标结构通常是视盘/杯，不一定是微小病灶。

本仓库状态：

- 本地已有 IDRID/DDR 数据适配。
- `matcher/data/dr.py`、`matcher/data/dr_interactive.py` 和 `matcher/Matcher_SAM*.py` 更接近“眼底病灶/局部区域”的实验代码。

### MedSAM-Agent

核心问题：把医学交互分割变成多轮 agent 决策：模型不只是一次性给点/框，而是观察结果、继续 refine prompt、减少无效操作。

本仓库状态：没有 agent/RL 训练代码。可以把它看作当前 demo 的高级自动化版本：我们的 demo 只做“一次 box prompt”，MedSAM-Agent 会自动规划多轮 prompt 和修正。

### MedSAM3

核心问题：把 SAM3 的 concept prompt 能力迁移到医学图像，让模型能用医学概念/文本描述定位器官、结构或病灶。

本仓库状态：没有 MedSAM3 代码。未来如果要升级，MedSAM3 适合替代“support mask/人工框”这类 prompt，让输入更接近医学语义。

## 三、本地代码和论文材料的对应关系

| 本地内容 | 对应思想/论文 | 说明 |
| --- | --- | --- |
| `segment_anything/` | SAM | promptable mask decoder |
| `dinov2/` | DINOv2 / Matcher | 通用视觉特征 |
| `matcher/` | Matcher | one-shot feature matching + SAM mask generation |
| `semantic_sam/` | Semantic-SAM | 更细粒度候选 mask |
| `matcher/data/MonuSeg.py` | 医学 nuclei segmentation | MonuSeg 数据接入 |
| `matcher/data/dr.py` | 医学眼底分割 | IDRID/DR 数据接入 |
| `demos/medical_sam_box_demo.py` | SAM / MedSAM 动机验证 | 直接看原始 SAM 在医学图像上的表现 |
| `D:\SAM\docs\papers\*.pdf` | 论文阅读材料 | 包含 SAM、SAM2、SAM3、Matcher、MedSAM-Agent、One-Prompt、UM-SAM、OSAM-Fundus 等 |

## 四、建议入门顺序

1. 先看 `demos/medical_sam_box_demo.py`：理解 prompt -> SAM mask 的最小闭环。
2. 再看 `matcher/Matcher_SAM.py`：理解 support mask 如何变成 query 图像上的点/框。
3. 看 `main_oss.py` 和 `matcher/data/dataset.py`：理解任务如何被组织成 few-shot episode。
4. 对照读 `D:\SAM\docs\papers\Matcher.pdf`：把代码里的 DINOv2、matching、SAM prompt 对上论文框架。
5. 再读 MedSAM / One-Prompt / MedSAM-Agent / MedSAM3：理解为什么医学领域要从“通用 SAM”继续做域适配、少提示和自动交互。

## 五、当前 demo 说明

当前已跑通：

```powershell
cd D:\SAM\code
python demos\medical_sam_box_demo.py --limit 4 --max-side 512
```

输出页面：

```text
D:\SAM\code\demo_outputs\medical_sam_box\index.html
```

这个 demo 是医学 SAM 入门验证，不是完整 MedSAM/MedSAM3 复现。它的价值是先把最小可运行链路打通，然后再逐步替换成 Matcher、MedSAM 或 SAM3 类方法。
