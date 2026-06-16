"""
plot_fg_bg_comparison.py - 简化版（服务器适用）

使用方法:
    python plot_fg_bg_comparison.py

输出:
    - dinov2_fg_bg_comparison.png
    - dinov2_fg_bg_comparison.pdf
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端（服务器必需）
import matplotlib.pyplot as plt
import numpy as np

# DINOv2 Small (原始)
dinov2_small_original = [
    0.4112, 0.4002, 0.4209, 0.4361, 0.3846, 0.4112, 0.4002, 0.4073, 0.4146, 0.3904,
    0.4278, 0.4361, 0.3899, 0.3899, 0.4073, 0.4361, 0.4112, 0.4197, 0.4002, 0.3620,
    0.4073, 0.4002, 0.4073, 0.4361, 0.4112, 0.4146, 0.4073, 0.4112, 0.3904, 0.3620,
    0.4146, 0.4274, 0.4112, 0.4073, 0.4274, 0.3899, 0.3899, 0.4002, 0.3904, 0.3899,
    0.4073, 0.4073, 0.4197, 0.4073, 0.4209, 0.4209, 0.4112, 0.3620, 0.4146, 0.3846
]

# DINOv2 Small Fine-tune 2
dinov2_small_ft2 = [
    0.6234, 0.2871, 0.3371, 0.6345, 0.3996, 0.6392, 0.3036, 0.1941, 0.5228, 0.2871,
    0.3036, 0.4426, 0.3996, 0.5228, 0.6345, 0.3167, 0.3167, 0.3371, 0.1941, 0.3996,
    0.2871, 0.6234, 0.5228, 0.5228, 0.6345, 0.6345, 0.3234, 0.3996, 0.6234, 0.3996,
    0.3996, 0.2871, 0.4426, 0.6392, 0.3234, 0.4426, 0.3036, 0.1941, 0.4426, 0.6345,
    0.6392, 0.6345, 0.3371, 0.4426, 0.6345, 0.4534, 0.3036, 0.6234, 0.2871, 0.6345
]

# DINOv2 Small Fine-tune 3
dinov2_small_ft3 = [
    0.6293, 0.2945, 0.3407, 0.4544, 0.3219, 0.2007, 0.6293, 0.3100, 0.4496, 0.3407,
    0.3100, 0.4544, 0.6332, 0.6332, 0.3318, 0.5267, 0.2945, 0.3318, 0.3219, 0.4544,
    0.6332, 0.2945, 0.6332, 0.3407, 0.4496, 0.6293, 0.4010, 0.3318, 0.4544, 0.2945,
    0.6451, 0.6451, 0.4544, 0.4544, 0.4544, 0.3407, 0.4496, 0.6332, 0.3407, 0.5267,
    0.3407, 0.3318, 0.4010, 0.3318, 0.2945, 0.4544, 0.3219, 0.2007, 0.3407, 0.5267
]

# DINOv2 Small Fine-tune 4
dinov2_small_ft4 = [
    0.6145, 0.2986, 0.3351, 0.4468, 0.2056, 0.5123, 0.4468, 0.3942, 0.6170, 0.3331,
    0.3351, 0.3331, 0.3105, 0.3331, 0.6170, 0.3331, 0.5123, 0.6287, 0.4464, 0.5123,
    0.6145, 0.6145, 0.3351, 0.6145, 0.6145, 0.6145, 0.5123, 0.6145, 0.6287, 0.6287,
    0.3351, 0.3942, 0.6287, 0.5123, 0.4464, 0.3105, 0.3351, 0.6287, 0.2056, 0.3351,
    0.3201, 0.5123, 0.6287, 0.3331, 0.3942, 0.3105, 0.6145, 0.4468, 0.2056, 0.3942
]

# DINOv2 Large (原始)
dinov2_large = [
    0.5701, 0.5853, 0.5724, 0.5650, 0.5711, 0.5523, 0.5609, 0.5724, 0.5711, 0.5701,
    0.5535, 0.5609, 0.5701, 0.5701, 0.5711, 0.5853, 0.5724, 0.5701, 0.5599, 0.5804,
    0.5804, 0.5701, 0.5588, 0.5609, 0.5523, 0.5535, 0.5588, 0.5609, 0.5853, 0.5853,
    0.5609, 0.5724, 0.5609, 0.5523, 0.5696, 0.5599, 0.5650, 0.5578, 0.5609, 0.5696,
    0.5588, 0.5711, 0.5588, 0.5804, 0.5696, 0.5804, 0.5650, 0.5650, 0.5578, 0.5650
]

# DINOv2 Large Fine-tune 1
dinov2_large_ft1 = [
    0.6226, 0.2793, 0.3873, 0.2793, 0.2793, 0.6401, 0.2793, 0.2793, 0.2793, 0.2793,
    0.2941, 0.2793, 0.3076, 0.3873, 0.6401, 0.4431, 0.3076, 0.4320, 0.6226, 0.3278,
    0.4431, 0.6306, 0.6226, 0.1901, 0.5149, 0.6401, 0.6306, 0.2793, 0.2941, 0.5149,
    0.2793, 0.1901, 0.4320, 0.3278, 0.3076, 0.5149, 0.4320, 0.2793, 0.2941, 0.1901,
    0.3278, 0.4431, 0.6401, 0.6226, 0.2941, 0.3146, 0.6226, 0.3873, 0.6226, 0.6401
]

# 样本编号
samples = np.arange(1, 51)

# 计算统计信息
mean_original = np.mean(dinov2_small_original)
mean_ft2 = np.mean(dinov2_small_ft2)
mean_ft3 = np.mean(dinov2_small_ft3)
mean_ft4 = np.mean(dinov2_small_ft4)
mean_dinov2_large = np.mean(dinov2_large)
mean_large_ft1 = np.mean(dinov2_large_ft1)

# 创建图表
plt.figure(figsize=(14, 7))

# 绘制六条折线
plt.plot(samples, dinov2_small_original, 
         marker='o', markersize=3, linewidth=2, 
         color='#1f77b4', label='DINOv2 Small (Original)', alpha=0.8)

plt.plot(samples, dinov2_small_ft2, 
         marker='s', markersize=3, linewidth=2, 
         color='#ff7f0e', label='DINOv2 Small Fine-tune 2', alpha=0.8)

plt.plot(samples, dinov2_small_ft3, 
         marker='^', markersize=3, linewidth=2, 
         color='#2ca02c', label='DINOv2 Small Fine-tune 3', alpha=0.8)

plt.plot(samples, dinov2_small_ft4, 
         marker='d', markersize=3, linewidth=2, 
         color='#d62728', label='DINOv2 Small Fine-tune 4', alpha=0.8)

# plt.plot(samples, dinov2_large, 
#          marker='p', markersize=3, linewidth=2, 
#          color='#9467bd', label='DINOv2 Large (Original)', alpha=0.8)

# plt.plot(samples, dinov2_large_ft1, 
#          marker='*', markersize=4, linewidth=2, 
#          color='#8c564b', label='DINOv2 Large Fine-tune 1', alpha=0.8)

# 添加平均线
plt.axhline(y=mean_original, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=mean_ft2, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=mean_ft3, color='#2ca02c', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=mean_ft4, color='#d62728', linestyle='--', linewidth=1, alpha=0.5)
# plt.axhline(y=mean_dinov2_large, color='#9467bd', linestyle='--', linewidth=1, alpha=0.5)
# plt.axhline(y=mean_large_ft1, color='#8c564b', linestyle='--', linewidth=1, alpha=0.5)

# 添加目标区域
plt.axhspan(0, 0.3, alpha=0.1, color='green', label='Target Zone (<0.3)')

# 图表设置
plt.xlabel('Sample ID', fontsize=12)
plt.ylabel('FG-BG Distance', fontsize=12)
plt.title('DINOv2 FG-BG Distance Comparison (Lower is Better)', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(loc='best', fontsize=9, ncol=2)
plt.ylim(0, 1)
plt.tight_layout()

# 保存图片
plt.savefig('dinov2_fg_bg_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('dinov2_fg_bg_comparison.pdf', bbox_inches='tight')
print("✓ Figures saved: dinov2_fg_bg_comparison.png and .pdf")

# 打印统计
print("\n" + "="*70)
print("Statistics Summary")
print("="*70)
print(f"DINOv2 Small (Original):     Mean = {mean_original:.4f}")
print(f"DINOv2 Small Fine-tune 2:    Mean = {mean_ft2:.4f}  | Δ = {mean_ft2-mean_original:+.4f}")
print(f"DINOv2 Small Fine-tune 3:    Mean = {mean_ft3:.4f}  | Δ = {mean_ft3-mean_original:+.4f}")
print(f"DINOv2 Small Fine-tune 4:    Mean = {mean_ft4:.4f}  | Δ = {mean_ft4-mean_original:+.4f}")
# print(f"DINOv2 Large (Original):     Mean = {mean_dinov2_large:.4f}")
# print(f"DINOv2 Large Fine-tune 1:    Mean = {mean_large_ft1:.4f}  | Δ = {mean_large_ft1-mean_dinov2_large:+.4f}")
print("="*70 + "\n")