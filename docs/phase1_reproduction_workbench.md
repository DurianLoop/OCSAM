# 第一阶段：复现工作台 (Reproduction Workbench)

## 目标

构建一个用于提示式（promptable）医疗图像和视频分割的统一复现工作台。第一阶段的目标尚不是提出新模型，而是让 SAM 家族的方法在一个统一接口下可运行、可比较且可切换。

核心目标是：

```text
predict(image_or_video, prompt, model_name, options) -> masks, boxes, scores, metadata
```

对于 Web 演示端，应表现为：

```text
选择模型 -> 选择场景 -> 选择提示类型 -> 运行 -> 检查结果
```

该工作台应至少支持 SAM、MedSAM、SAM2、Matcher 和 SAM3。候选扩展包括 MedSAM2、Medical SAM2、类 MedSAM3 变体、SAM-Med2D 以及其他如果能适配统一接口协议的医疗 SAM 衍生模型。

## 为什么这是第一阶段

这一阶段对应项目计划的第一部分：

```text
第 1 阶段：复现工作台
在统一的预测接口下整合 SAM, MedSAM, SAM2, Matcher 和 SAM3。

第 2 阶段：基准测试
在相同的数据和提示协议上运行每个模型，识别失效模式。

第 3 阶段：核心创新
优先考虑医疗提示自动化、SAM2 记忆选择和轻量化适配器。

第 4 阶段：论文发布
通过结构化医疗提示和跨切片记忆选择，声明在准确性、稳定性和交互效率上的提升。
```

第一阶段应产出一个可工作的平台，而不仅仅是孤立的脚本。

## 本地数据边界

当前的本地数据集已经定义了一个合理的 2D 医疗任务边界。

### IDRID

路径：

```text
D:\SAM\code\datasets\IDRID
```

观测到的本地结构：

```text
DR_Training_Set/Fundus Images     54 张 jpg 图像
DR_Training_Set/Combined Masks    54 张 png 掩码
DR_Testing_Set/Fundus Images      27 张 jpg 图像
DR_Testing_Set/Combined Masks     27 张 png 掩码
```

建议任务：

```text
眼底糖尿病视网膜病变病灶分割 (Fundus diabetic retinopathy lesion segmentation)
```

初始协议：

- 二值前景：将所有非背景的病灶颜色合并为一个病灶掩码。
- 可选高级协议：如果颜色语义可以恢复并验证，则按颜色将掩码拆分为病灶子类。
- 提示类型：框（box）、正/负点（positive/negative points）、文本提示（如 `lesion`, `red lesion`, `bright lesion`, `exudate`）。

为什么重要：

- 目标微小。
- 低对比度。
- 前景稀疏。
- 具有临床意义的失效模式。

### MonuSeg

路径：

```text
D:\SAM\code\datasets\MonuSeg
```

观测到的本地结构：

```text
kmms/images                 82 张图像
kmms/masks                  82 张掩码
kmms_test/kmms_test/images  58 张图像
kmms_test/kmms_test/masks   58 张掩码
```

建议任务：

```text
组织病理学细胞核分割 (Histopathology nuclei segmentation)
```

初始协议：

- 二值前景：所有细胞核标签均为前景。
- 可选高级协议：保留实例标签，用于实例级细胞核评估。
- 提示类型：框、正/负点、参考掩码（reference mask）、文本提示（如 `nucleus`, `nuclei`, `cell nucleus`）。

为什么重要：

- 高密度小物体。
- 边界接触和重叠。
- 实例分离对通用 SAM 来说非常困难。

### 非医疗辅助数据集

本地的 `COCO2014` 和 `FSS-1000` 文件夹对于 Matcher 的正确性检查和通用少样本（few-shot）分割很有用，但不应被视为医疗主任务。

## 模型矩阵 (Model Matrix)

### SAM

角色：

```text
通用提示式分割基准 (Generic promptable segmentation baseline)
```

本地状态：

- 现有本地权重：`D:\SAM\assets\checkpoints\sam_vit_b.pth`
- 现有代码：`D:\SAM\code\segment_anything`
- 现有演示工作：`D:\SAM\code\demos\medical_sam_click_app.py`

提示支持：

- 点提示。
- 框提示。
- 自动掩码（Automatic masks）。
- 无原生文本提示。

适配器目标：

```python
predict_sam(image, prompt) -> Prediction
```

实现说明：

- 直接使用 SAM 的点/框提示。
- 文本提示只能是启发式的，除非配合检测器或 SAM3 风格的开集模型。
- 保持 SAM 作为纯净的基准。

### MedSAM

角色：

```text
围绕框提示构建的医疗图像分割基准
```

提示支持：

- 主要是框提示。
- 当目标框质量较好时，对医疗结构的支持通常很强。

适配器目标：

```python
predict_medsam(image, box_prompt) -> Prediction
```

第 1 阶段任务：

- 在独立文件夹下添加或克隆 MedSAM 源码。
- 下载/定位权重文件。
- 编写适配器，接收与 Web App 相同的归一化框提示。
- 首先在 IDRID 和 MonuSeg 上使用基于真值（GT）衍生的框进行验证。

重要对比：

```text
在完全相同的 GT 框提示下，比较 SAM box vs MedSAM box
```

### SAM2

角色：

```text
图像/视频提示式分割和传播基准
```

提示支持：

- 图像点/框提示。
- 视频提示和掩码传播。

适配器目标：

```python
predict_sam2_image(image, prompt) -> Prediction
predict_sam2_video(video_or_frames, prompt_on_frame) -> VideoPrediction
```

第 1 阶段任务：

- 克隆或集成官方 SAM2 代码。
- 下载权重。
- 先制作图像适配器。
- 后制作视频适配器。
- 对于医疗用途，准备一种“伪视频”模式，将有序切片或相关的图像序列视为帧。

重要对比：

```text
SAM 图像提示 vs SAM2 图像提示
SAM2 在简单视频或切片序列上的单帧提示传播
```

### Matcher

角色：

```text
单样本/参考掩码分割基准 (One-shot/reference-mask segmentation baseline)
```

本地状态：

- 现有源码：`D:\SAM\code\matcher`, `D:\SAM\Matcher`
- 现有医疗加载器：`dr.py`, `dr_interactive.py`, `MonuSeg.py`

提示支持：

- 参考图像 + 参考掩码。
- 目标图像。
- 设计上不是常规的点选/框选模型。

适配器目标：

```python
predict_matcher(reference_image, reference_mask, target_image, options) -> Prediction
```

第 1 阶段任务：

- 封装现有的 Matcher 代码，而不是重写它。
- 构建一个 Web 场景，当切换到 Matcher 时，提示面板转变为参考图像和参考掩码的选择。
- 使用 MonuSeg 和 IDRID 作为首批医疗单样本案例。

重要对比：

```text
带有目标提示的 SAM/MedSAM vs 带有参考示例的 Matcher
```

### SAM3

角色：

```text
开集概念分割和文本提示基准
```

本地状态：

- 源码已克隆至 `D:\SAM\sam3`
- 存在本地 Windows 兼容性补丁（用于无 Triton 环境的 EDT 回退）。
- 存在本地脚本：`D:\SAM\sam3\examples\local_sam3_image_demo.py`
- Hugging Face 门控访问（gated access）目前被拦截。

提示支持：

- 文本提示。
- 框提示。
- 根据模型变体支持图像/视频使用。

适配器目标：

```python
predict_sam3_image(image, text_prompt=None, box_prompt=None) -> Prediction
```

第 1 阶段任务：

- 解决权重访问问题。
- 如果 ModelScope 镜像提供官方文件且无 HF 门控摩擦，优先选择。
- 在 IDRID 和 MonuSeg 示例上运行文本提示。
- 比较文本提示质量与启发式 SAM 文本模式以及直接的点/框提示。

重要对比：

```text
SAM3 文本提示 vs SAM/MedSAM 框提示 vs Matcher 参考提示
```

## 候选医疗扩展

不要一次性添加所有扩展。仅在主要的五模型矩阵稳定后才添加。

候选列表：

- MedSAM2 / Medical SAM2：适用于 3D 或视频风格的医疗传播。
- SAM-Med2D：如果我们需要 2D 医疗 SAM 基准。
- 类 MedSAM3 变体：仅当它们有公开代码和权重时。
- 眼底或病理学的专用论文模型：后期作为特定任务的基准有用，不作为首个平台骨干。

在第 1 阶段，每个扩展必须满足：

```text
它能在本地运行吗？
它能接受我们的某种提示类型吗？
它能以通用格式返回掩码/框/分数吗？
它能至少在 IDRID 或 MonuSeg 上运行吗？
```

## 统一适配器协议 (Unified Adapter Contract)

每个模型都应实现相同的高层适配器。

```python
class Prompt:
    type: str  # point, box, text, reference, mask, video_point, video_text
    points: list[tuple[float, float, int]] | None
    box_xyxy: tuple[float, float, float, float] | None
    text: str | None
    reference_image: Any | None
    reference_mask: Any | None
    frame_index: int | None


class Prediction:
    masks: list[Any]
    boxes_xyxy: list[tuple[float, float, float, float]]
    scores: list[float]
    overlay: Any
    metadata: dict
```

每个适配器应暴露：

```python
load_model(config) -> ModelHandle
predict(handle, sample, prompt, options) -> Prediction
supports() -> dict
```

`supports()` 方法应告知 UI 显示哪些提示控件。

示例：

```json
{
  "image": true,
  "video": false,
  "point": true,
  "box": true,
  "text": false,
  "reference": false
}
```

## Web 工作台设计

Web 页面不应是每个模型的单独演示。它应该是一个具有模型感知控件的统一工作台。

### 左侧面板

模型选择器：

```text
SAM
MedSAM
SAM2
Matcher
SAM3
MedSAM2
```

场景选择器：

```text
IDRID 简单病灶
IDRID 困难微小病灶
MonuSeg 稀疏细胞核
MonuSeg 稠密细胞核
通用自然图像
视频 简单
视频 困难
```

提示模式选择器：

```text
点 (Point)
框 (Box)
文本 (Text)
参考图像/掩码 (Reference image/mask)
掩码精细化 (Mask refinement)
视频帧提示 (Video frame prompt)
```

提示控件应随所选模型而变化。

示例：

- SAM：点、框、自动掩码。
- MedSAM：框。
- SAM2 图像：点、框。
- SAM2 视频：在选定帧上点/框，然后传播。
- Matcher：参考图像 + 参考掩码 + 目标图像。
- SAM3：文本、框，可能包括视频文本。

### 中间面板

输入查看器：

- 2D 图像画布。
- 视频的帧查看器和时间轴。
- 点、框和参考掩码的提示覆盖层。

### 右侧面板

输出查看器：

- 掩码覆盖图。
- 单个实例列表。
- 分数。
- 运行时长。
- GPU 显存（如果可用）。
- 提示次数。
- 保存结果按钮。

### 底部面板

对比栏：

- 相同样本、相同提示下的多模型输出。
- 如果真值可用，显示 Dice/IoU 的简表。
- 用于手动注释的失效说明文本框。

## 场景分类 (Scene Taxonomy)

第一阶段就应该编码“容易”和“困难”案例，因为 Web 演示应该教会我们模型在哪里会失败。

### IDRID 场景

容易：

- 大型明亮病灶。
- 清晰的高对比度病灶区域。

困难：

- 微小红色病灶。
- 前景稀疏。
- 靠近血管的病灶。
- 低对比度、光照不均匀。

### MonuSeg 场景

容易：

- 孤立的细胞核。
- 染色清晰。

困难：

- 密集接触的细胞核。
- 边界模糊。
- 混合染色。
- 许多微小实例。

### 视频场景

简单：

- 单个物体，外观稳定。
- 短片段。

困难：

- 遮挡。
- 外观变化。
- 多个相似物体。
- 来自切片的医疗伪视频。

## 提示协议 (Prompt Protocols)

从一开始就使用固定的提示协议。否则第 2 阶段的基准测试将不公平。

### 点协议

用于真值辅助评估：

- 在掩码中心取一个正点。
- 可选：从邻近背景采样负点。
- 多次点击协议：每轮在最大误差区域添加一个点。

用于手动演示：

- 左键点击：正点。
- 右键点击：负点。

### 框协议

用于评估：

- 使用真值边界框。
- 可选：添加随机抖动（jitter）以模拟不完美的用户输入框。

用于手动演示：

- 在画布上拖拽框。

### 文本协议

用于 SAM3：

- IDRID：`lesion`, `red lesion`, `bright lesion`, `exudate`
- MonuSeg：`nucleus`, `cell nucleus`, `nuclei`

用于 SAM：

- 不要假装 SAM 有原生文本支持。
- 如果存在文本模式，将其标记为“启发式”或“检测器辅助”。

### 参考协议

用于 Matcher：

- 选取一张支持图像（support image）和支持掩码。
- 在目标图像上预测。
- 保持基准测试中的支持-目标对固定。

## ModelScope SAM3 权重计划

用户找到的 ModelScope 页面是：

```text
https://www.modelscope.cn/models/facebook/sam3/files
```

推荐下载途径：

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -m pip install modelscope
```

然后使用 Python：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:ALL_PROXY=""

D:\SAM\conda_envs\sam_gpu\python.exe -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('facebook/sam3', local_dir=r'D:\SAM\assets\checkpoints\sam3_modelscope')"
```

预期查找的文件：

```text
config.json
sam3.pt
```

如果 ModelScope CLI 可用，此形式也可能奏效（取决于安装版本）：

```powershell
modelscope download --model facebook/sam3 --local_dir D:\SAM\assets\checkpoints\sam3_modelscope
```

下载后，使用本地权重运行 SAM3 演示：

```powershell
cd D:\SAM\sam3

$env:PYTHONPATH="D:\SAM\sam3\.deps_runtime;D:\SAM\sam3"

D:\SAM\conda_envs\sam_gpu\python.exe -B examples\local_sam3_image_demo.py `
  --image assets\images\test_image.jpg `
  --prompt person `
  --checkpoint D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt `
  --out-dir outputs
```

注意：

- 官方 Hugging Face 仓库 `facebook/sam3` 是受限的，且当前账户返回了审批错误。
- 如果 ModelScope 镜像了相同文件，它可以绕过 HF 审批路径。
- 在将权重路径接入 Web 工作台之前，请验证文件名。

## 实施里程碑

### 里程碑 1：清单与注册表 (Inventory And Registry)

产出物：

- `model_registry.json` 或 Python 注册表，列出所有适配器。
- 用于 IDRID、MonuSeg、通用图像和视频示例的数据集注册表。
- 每个模型的能力矩阵。

完成定义：

```text
Web App 可以在加载模型之前，列出模型、场景和合法的提示模式。
```

### 里程碑 2：通用数据和提示对象

产出物：

- 共享的 `Sample` 对象。
- 共享的 `Prompt` 对象。
- 共享的 `Prediction` 对象。
- 用于图像缩放和坐标转换的工具函数。

完成定义：

```text
SAM 和 MedSAM 可以消耗相同的框提示对象。
```

### 里程碑 3：前两个适配器

从以下开始：

- SAM。
- MedSAM。

完成定义：

```text
两个模型都能在同一个 Web UI 中运行相同的 IDRID 图像和 MonuSeg 图像。
```

### 里程碑 4：Matcher 适配器

产出物：

- 参考图像选择器。
- 参考掩码选择器。
- 目标图像选择器。
- Matcher 输出转换为通用 `Prediction`。

完成定义：

```text
Matcher 可以在一个 MonuSeg 支持-目标对和一个 IDRID 支持-目标对上运行。
```

### 里程碑 5：SAM2 适配器

产出物：

- 图像模式。
- 视频或伪视频模式。
- 帧提示选择器。

完成定义：

```text
SAM2 可以运行图像提示，并至少演示一个视频/伪视频传播。
```

### 里程碑 6：SAM3 适配器

产出物：

- 文本提示。
- 框提示（如果稳定）。
- 本地权重路径支持。

完成定义：

```text
SAM3 可以从本地权重运行，并通过通用接口返回文本提示掩码。
```

### 里程碑 7：对比视图

产出物：

- 运行单个模型。
- 在相同样本/提示下运行选定的多个模型。
- 并排显示输出。

完成定义：

```text
用户可以在一个页面中直观地比较 SAM, MedSAM, SAM2, Matcher 和 SAM3 的结果。
```

## 建议目录结构

```text
D:\SAM\code
  adapters\
    base.py
    sam_adapter.py
    medsam_adapter.py
    sam2_adapter.py
    matcher_adapter.py
    sam3_adapter.py
  workbench\
    app.py
    registry.py
    samples.py
    prompts.py
    metrics.py
    static\
    templates\
  demo_outputs\
    workbench\
```

保持模型源代码库独立：

```text
D:\SAM\code          当前主工作空间
D:\SAM\Matcher       纯净的 Matcher 引用
D:\SAM\sam3          SAM3 官方源码克隆
D:\SAM\assets        权重和大文件
```

## 第 1 阶段验收标准

当满足以下条件时，第 1 阶段完成：

- Web 页面可以在至少 SAM, MedSAM, SAM2, Matcher, 和 SAM3 之间切换。
- 每个模型都声明其支持的提示类型。
- IDRID 和 MonuSeg 作为内置场景可用。
- 每个医疗数据集至少存在一个简单场景和一个困难场景。
- 结果以通用格式返回。
- 输出结果可以保存。
- 运行时错误在 UI 中清晰显示，而不是导致服务器崩溃。
- 同一个提示可以在兼容的模型之间重放。

如果存在以下情况，第 1 阶段未完成：

- 每个模型仍然使用单独的临时 Demo。
- 提示坐标在每个模型中处理方式不同且未进行转换。
- 结果无法并排比较。
- SAM 的文本模式被呈现为原生 SAM 文本分割。
- SAM3 依赖运行时的实时网络下载。

## 立即执行的后续行动

1. 如果可以访问，从 ModelScope 下载 SAM3 权重。
2. 添加正式的适配器接口（interface）。
3. 将当前的 SAM 点击/框选/文本 Demo 重构为第一个工作台 App。
4. 为 IDRID 和 MonuSeg 添加数据集场景清单（manifests）。
5. 首先添加 SAM 适配器和 MedSAM 适配器。
6. 在 Web 提示面板支持参考图像/掩码后添加 Matcher。
7. 添加 SAM2 图像模式，然后是视频/伪视频模式。
8. 仅在本地权重加载工作后才添加 SAM3。

## 当前实现进度更新

截至本轮实现，第一阶段已经从“单文件 demo 原型”推进到“带统一契约与注册表的工作台原型”。

已补齐：

- 新增统一适配器契约：`D:\SAM\code\adapters\base.py`
  - `Prompt`
  - `Prediction`
  - `ModelSupport`
  - `BasePromptAdapter`
- 新增模型能力注册表：`D:\SAM\code\workbench\registry.py`
  - SAM
  - MedSAM
  - SAM2
  - Matcher
  - SAM3
- 新增样本与场景注册表：`D:\SAM\code\workbench\samples.py`
  - IDRID easy/hard 场景
  - MonuSeg easy/hard 场景
  - SAM2 video 场景
- Web API 新增：
  - `GET /models`
  - `GET /samples`
  - `GET /examples` 返回 dataset / scene / difficulty
  - `GET /videos` 返回 dataset / scene / difficulty
- Web 推理返回新增结构化 `prediction` 字段：
  - `masks`
  - `boxes_xyxy`
  - `scores`
  - `metadata`
- Web UI 新增结果下载按钮。
- 切换兼容模型时保留当前 prompt，支持同一提示在兼容模型间重放。
- 新增独立 adapter 骨架文件：
  - `D:\SAM\code\adapters\sam_adapter.py`
  - `D:\SAM\code\adapters\medsam_adapter.py`
  - `D:\SAM\code\adapters\sam2_adapter.py`
  - `D:\SAM\code\adapters\matcher_adapter.py`
  - `D:\SAM\code\adapters\sam3_adapter.py`
- Web 推理返回中的 `prediction` 字段已经包含可序列化 mask PNG、box、score 和 metadata。
- 新增 `POST /compare_segment`，支持同一 image prompt 在 SAM / SAM2 / SAM3 / MedSAM 间并排运行。
- Web UI 新增 Compare 按钮和并排结果网格。
- Matcher 支持 reference mask 上传；未上传 mask 时退回 reference box 生成矩形支持掩码。

仍未完全补齐：

- 还没有 GT 驱动的 Dice/IoU benchmark。
- 独立 adapter 文件目前是能力声明和迁移骨架，运行时仍由 `demos/medical_sam_click_app.py` 中的现有 wrapper 承担；后续可进一步把实际加载和 predict 逻辑迁入这些 adapter。

因此当前状态为：

```text
Phase 1 reproduction workbench: completed for interactive reproduction
Phase 1 benchmark-ready evaluation: pending GT Dice/IoU layer, which belongs to Phase 2
```
