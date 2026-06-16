# SAM Medical Segmentation Workspace

这是医学图像分割复现与入门实验工作区。当前重点是围绕 SAM、MedSAM、Matcher、SAM2/SAM3 等模型，完成源码拉取、模型准备、GPU 环境配置、批量评测和 demo 展示。

## 快速入口

激活 GPU 环境：

```powershell
cd D:\SAM
.\activate_sam_env.ps1
```

运行当前批量评测：

```powershell
cd D:\SAM\code
python demos\medical_sam_batch_eval.py --dataset all --max-side 512 --out-dir demo_outputs\medical_sam_batch_eval_full_v2
```

查看最新批量结果：

```text
D:\SAM\code\demo_outputs\medical_sam_batch_eval_full_v2\README.md
D:\SAM\code\demo_outputs\medical_sam_batch_eval_full_v2\index.html
D:\SAM\code\demo_outputs\medical_sam_batch_eval_full_v2\metrics.csv
D:\SAM\code\demo_outputs\medical_sam_batch_eval_full_v2\summary.json
```

## 目录说明

| 路径 | 作用 |
|---|---|
| `code` | 当前主要工作仓库，包含改造后的 Matcher/SAM 医学分割代码、数据、demo、批量评测输出 |
| `Matcher` | Matcher 官方源码干净副本，用来对照论文和原始实现 |
| `assets/checkpoints` | 统一模型权重目录，保存 SAM、DINOv2、Matcher/IDRID 相关权重 |
| `conda_envs/sam_gpu` | GPU 版 conda 环境，PyTorch CUDA 可用 |
| `docs/papers` | 已下载论文 PDF |
| `docs/workspace` | 环境、布局和工作区说明文档 |
| `docs/setup_legacy` | 早期安装记录和旧 requirements，作为历史参考 |
| `.cache` | 工作区本地缓存，主要给 matplotlib 使用 |

`code\models` 和 `Matcher\models` 现在是 junction，指向同一份 `assets\checkpoints`。这样代码里原来的 `models/sam_vit_b.pth` 路径还能用，同时避免三份大模型重复占空间。

更详细的说明见：

```text
D:\SAM\docs\workspace\WORKSPACE_GUIDE.md
```
