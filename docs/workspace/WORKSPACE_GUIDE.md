# 工作区结构说明

本文档说明 `D:\SAM` 里每个主要目录的作用、当前实验做到哪一步，以及后续推荐怎么推进。

## 1. 顶层结构

```text
D:\SAM
├── activate_sam_env.ps1
├── assets
│   └── checkpoints
├── code
├── conda_envs
│   └── sam_gpu
├── docs
│   ├── papers
│   ├── setup_legacy
│   └── workspace
├── Matcher
└── .cache
```

## 2. 主要目录

### `code`

这是当前主要工作仓库。我们在这里做医学图像分割 demo、批量评测、代码兼容性修复和后续实验扩展。

重要子目录：

| 路径 | 作用 |
|---|---|
| `demos` | 入门 demo 和批量评测脚本 |
| `datasets` | 当前已放入的医学图像数据，例如 MonuSeg、IDRID |
| `demo_outputs` | demo 和批量评测输出 |
| `docs` | 代码与论文关系说明 |
| `matcher` | Matcher 核心代码 |
| `segment_anything` | SAM 原始分割模块 |
| `semantic_sam` | Semantic-SAM 相关代码 |
| `models` | junction，实际指向 `D:\SAM\assets\checkpoints` |

当前最重要的脚本：

```text
D:\SAM\code\demos\medical_sam_batch_eval.py
D:\SAM\code\demos\medical_sam_box_demo.py
```

当前最重要的结果：

```text
D:\SAM\code\demo_outputs\medical_sam_batch_eval_full_v2\
```

这批结果使用 SAM ViT-B + ground-truth bbox prompt，在 163 张本地医学图像上完成了批量测试。

### `Matcher`

这是 Matcher 官方仓库的干净副本，用途是：

- 对照论文原始实现；
- 对照我们在 `code` 里做的改动；
- 避免在实验改造中丢失官方 baseline。

它的 `models` 目录也是 junction，指向 `D:\SAM\assets\checkpoints`。

### `assets/checkpoints`

这是统一的模型权重目录。之前权重在三个地方重复保存：

```text
D:\SAM\pretrained_models
D:\SAM\code\models
D:\SAM\Matcher\models
```

现在已经合并为一份：

```text
D:\SAM\assets\checkpoints
```

当前包含：

| 文件 | 作用 |
|---|---|
| `sam_vit_b.pth` | SAM ViT-B，当前 demo 和批量评测使用 |
| `sam_vit_h_4b8939.pth` | SAM ViT-H，大模型版本 |
| `dinov2_vitl14_pretrain.pth` | DINOv2 ViT-L/14，Matcher 相关 |
| `swint_only_sam_many2many.pth` | Matcher/Semantic-SAM 相关权重 |
| `best_backbone_idrid_finetuned.pth` | IDRID 相关微调权重 |

为了不破坏已有代码路径，下面两个目录是 junction：

```text
D:\SAM\code\models -> D:\SAM\assets\checkpoints
D:\SAM\Matcher\models -> D:\SAM\assets\checkpoints
```

### `conda_envs`

这里放 conda 环境。当前可用 GPU 环境是：

```text
D:\SAM\conda_envs\sam_gpu
```

不要随意移动这个目录。conda 环境通常不是完全可搬家的，移动后 Python、pip、脚本路径可能失效。

激活方式：

```powershell
cd D:\SAM
.\activate_sam_env.ps1
```

环境细节见：

```text
D:\SAM\docs\workspace\ENVIRONMENT.md
```

### `docs/papers`

这里放论文 PDF。当前包括 SAM、SAM2、SAM3、Matcher、One-Prompt、MedSAM-Agent、OSAM-Fundus、UM-SAM 等资料。

这些论文的作用是帮助建立路线：

```text
SAM -> 医学 SAM/MedSAM -> Matcher/one-shot segmentation -> SAM2/SAM3 -> 医学 agent/自动化分割
```

### `docs/workspace`

这里放工作区文档：

| 文件 | 作用 |
|---|---|
| `ENVIRONMENT.md` | GPU 环境、验证命令、运行方式 |
| `WORKSPACE_LAYOUT_OLD.md` | 整理前的旧布局记录 |
| `WORKSPACE_GUIDE.md` | 当前这份总说明 |

### `docs/setup_legacy`

这里放早期安装记录和旧 requirements。它不是当前主环境配置来源，只作为排查或追溯参考。

### `.cache`

本地缓存目录。目前主要用于 matplotlib，避免它往用户目录写缓存时遇到权限问题。

## 3. 当前已经完成的工作

已完成：

- 拉取/准备 Matcher 官方源码；
- 保留一个工作版源码目录 `code`；
- 下载并统一整理模型权重；
- 配置 GPU conda 环境；
- 验证 RTX 5090 可用；
- 跑通 SAM ViT-B 医学图像 box-prompt demo；
- 新增批量评测脚本；
- 对 MonuSeg 和 IDRID 做 163 张批量测试；
- 输出 HTML、CSV、JSON、README 结果包。

当前批量结果摘要：

| Dataset / split | N | Mean IoU | Mean Dice |
|---|---:|---:|---:|
| MonuSeg nuclei / kmms | 82 | 0.308 | 0.443 |
| IDRID diabetic retinopathy lesions / train | 54 | 0.059 | 0.110 |
| IDRID diabetic retinopathy lesions / test | 27 | 0.063 | 0.118 |

## 4. 为什么这样整理

整理原则是：

- 不移动 conda 环境，避免环境失效；
- 不移动主源码仓库，避免刚跑通的命令失效；
- 合并确定重复的大模型，减少磁盘占用；
- 把论文和安装记录从根目录移走，让根目录只保留入口级内容；
- 保留官方源码副本，方便区分官方 baseline 和实验改造版。

## 5. 后续规划

建议后续按这个顺序推进：

1. 固化当前 SAM ViT-B baseline  
   当前 `medical_sam_batch_eval_full_v2` 就是第一版可复现实验结果。

2. 加 MedSAM 或医学 SAM 权重  
   在同一批 MonuSeg/IDRID 上跑相同协议，和原始 SAM 对比。

3. 跑 Matcher one-shot pipeline  
   从 “GT box prompt” 进入 “参考图 + 参考 mask” 的 one-shot segmentation。

4. 做交互式 Gradio demo  
   支持上传图像、选择模型、框选/参考样本、输出 mask 和指标。

5. 写入门报告  
   把论文关系、代码结构、实验结果和失败案例整理成一份学习文档。

## 6. 常用命令

激活环境：

```powershell
cd D:\SAM
.\activate_sam_env.ps1
```

验证 GPU：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

运行批量评测：

```powershell
cd D:\SAM\code
python demos\medical_sam_batch_eval.py --dataset all --max-side 512 --out-dir demo_outputs\medical_sam_batch_eval_full_v2
```

只跑 MonuSeg：

```powershell
python demos\medical_sam_batch_eval.py --dataset monuseg --max-side 512 --out-dir demo_outputs\monuseg_sam_eval
```

只跑 IDRID test：

```powershell
python demos\medical_sam_batch_eval.py --dataset idrid_test --max-side 512 --out-dir demo_outputs\idrid_test_sam_eval
```
