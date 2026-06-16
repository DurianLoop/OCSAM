# SAM GPU Conda Environment

Environment path:

```text
D:\SAM\conda_envs\sam_gpu
```

Shared checkpoint path:

```text
D:\SAM\assets\checkpoints
```

The repository-local `models` directories are junctions to this shared checkpoint path:

```text
D:\SAM\code\models -> D:\SAM\assets\checkpoints
D:\SAM\Matcher\models -> D:\SAM\assets\checkpoints
```

Activate in PowerShell:

```powershell
cd D:\SAM
.\activate_sam_env.ps1
```

Run the medical SAM demo:

```powershell
cd D:\SAM\code
python demos\medical_sam_box_demo.py --limit 4 --max-side 512
```

Run the same command without activating:

```powershell
$env:MPLCONFIGDIR="D:\SAM\.cache\matplotlib_gpu"
D:\SAM\conda_envs\sam_gpu\python.exe demos\medical_sam_box_demo.py --limit 4 --max-side 512
```

Installed core versions:

- Python 3.11.15
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- CUDA runtime used by PyTorch: 12.8
- Gradio 3.24.1
- gradio-client 0.0.8
- timm 0.9.16
- matplotlib 3.11.0
- scipy / scikit-image / scikit-learn / pycocotools / tensorboardX

Verification:

```powershell
cd D:\SAM
.\activate_sam_env.ps1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected result in this workspace:

```text
2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5090 D v2
```

GPU demo smoke test:

```powershell
cd D:\SAM\code
python demos\medical_sam_box_demo.py --limit 1 --max-side 384 --out-dir demo_outputs\gpu_check_medical_sam_box
```

The demo should print `Using device: cuda:0` and write:

```text
D:\SAM\code\demo_outputs\gpu_check_medical_sam_box\index.html
```

Notes:

- The machine reports an RTX 5090 through `nvidia-smi`; this environment verifies `torch.cuda.is_available() == True`.
- The previous CPU-only environment still exists at `D:\SAM\conda_envs\sam`, but `activate_sam_env.ps1` now activates the GPU environment.
- The GPU PyTorch package follows the official PyTorch Windows pip route with CUDA 12.8 wheels.
- The official project expects Linux/macOS for full Detectron2/Semantic-SAM support. In this Windows workspace, Detectron2-backed LVIS/PACO-Part registration is optional so the medical and common Matcher paths can import.
- A small local `cv2.py` compatibility layer is present in `code/` and `Matcher/` because the conda OpenCV DLL build failed to import on this machine.
