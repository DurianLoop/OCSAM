$env:MPLCONFIGDIR = "D:\SAM\.cache\matplotlib_gpu"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

. D:\conda\shell\condabin\conda-hook.ps1
conda activate D:\SAM\conda_envs\sam_gpu

Write-Host "SAM GPU environment activated: D:\SAM\conda_envs\sam_gpu"
Write-Host "Use: cd D:\SAM\code"
Write-Host "Demo: python demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860"
Write-Host "Note: this helper is for the local D:\SAM machine. New users should follow README.md to create their own conda environment."
