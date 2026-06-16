import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import os


class DINOv2SlidingWindowWrapper:
    """
    使用滑动窗口策略提取高分辨率DINOv2特征（串行处理版本）
    
    输入: 1036x1036图像（已上采样）
    输出: 74x74 patch grid特征 (5476个patches)
    
    策略: 
    - 9个518x518窗口（3x3网格，50%重叠）
    - 串行处理每个窗口（节省显存）
    - 对重叠区域特征取平均
    """
    
    def __init__(self, encoder, patch_size=14, window_size=518, target_size=1036):
        """
        Args:
            encoder: DINOv2模型
            patch_size: DINOv2的patch大小（默认14）
            window_size: 滑动窗口大小（默认518，DINOv2最优输入尺寸）
            target_size: 输入图像尺寸（默认1036，2倍window_size）
        """
        self.encoder = encoder
        self.patch_size = patch_size
        self.window_size = window_size
        self.target_size = target_size
        
        # 计算窗口布局（3x3网格，50%重叠）
        self.stride = window_size // 2  # 259
        self.num_windows_per_axis = 3
        self.total_windows = 9
        
        # 计算最终patch grid大小
        self.final_patch_grid = target_size // patch_size  # 74
        self.total_patches = self.final_patch_grid ** 2  # 5476
        
        # 窗口内patch grid大小
        self.window_patch_grid = window_size // patch_size  # 37
        
        print(f"\n{'='*70}")
        print(f"DINOv2 Sliding Window Wrapper Initialized (Sequential Mode)")
        print(f"{'='*70}")
        print(f"Input size:       {target_size}x{target_size} pixels")
        print(f"Window size:      {window_size}x{window_size} pixels")
        print(f"Stride:           {self.stride} pixels (50% overlap)")
        print(f"Windows:          {self.total_windows} (3x3 grid)")
        print(f"Patch size:       {patch_size}x{patch_size} pixels")
        print(f"Output grid:      {self.final_patch_grid}x{self.final_patch_grid} patches")
        print(f"Total patches:    {self.total_patches}")
        print(f"Feature dim:      {encoder.embed_dim}")
        print(f"Processing mode:  Sequential (one window at a time)")
        print(f"{'='*70}\n")
        
    def get_window_positions(self) -> List[Tuple[int, int, int, int]]:
        """
        生成9个窗口的位置坐标 (y1, y2, x1, x2)
        
        窗口布局 (像素坐标):
        ┌─────────────┬─────────────┬─────────────┐
        │  Window 0   │  Window 1   │  Window 2   │
        │  [0:518,    │  [0:518,    │  [0:518,    │
        │   0:518]    │   259:777]  │   518:1036] │
        ├─────────────┼─────────────┼─────────────┤
        │  Window 3   │  Window 4   │  Window 5   │
        │ [259:777,   │ [259:777,   │ [259:777,   │
        │   0:518]    │   259:777]  │   518:1036] │
        ├─────────────┼─────────────┼─────────────┤
        │  Window 6   │  Window 7   │  Window 8   │
        │ [518:1036,  │ [518:1036,  │ [518:1036,  │
        │   0:518]    │   259:777]  │   518:1036] │
        └─────────────┴─────────────┴─────────────┘
        
        Returns:
            List of (y1, y2, x1, x2) for each window
        """
        positions = []
        
        for i in range(self.num_windows_per_axis):  # 行: 0, 1, 2
            for j in range(self.num_windows_per_axis):  # 列: 0, 1, 2
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = y1 + self.window_size
                x2 = x1 + self.window_size
                
                # 边界处理：确保不超出1036
                if y2 > self.target_size:
                    y2 = self.target_size
                    y1 = y2 - self.window_size
                if x2 > self.target_size:
                    x2 = self.target_size
                    x1 = x2 - self.window_size
                    
                positions.append((y1, y2, x1, x2))
        
        return positions
    
    def extract_window_features(self, img: torch.Tensor, window_pos: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        提取单个窗口的DINOv2特征
        
        Args:
            img: 输入图像 [B, C, 1036, 1036]
            window_pos: 窗口位置 (y1, y2, x1, x2)
            
        Returns:
            特征 [B, 1369, feature_dim]，已去除CLS token
        """
        y1, y2, x1, x2 = window_pos
        window = img[:, :, y1:y2, x1:x2]
        
        # 确保窗口大小正确
        assert window.shape[-2:] == (self.window_size, self.window_size), \
            f"Window shape {window.shape[-2:]} != ({self.window_size}, {self.window_size})"
        
        # 提取特征（去除CLS token）
        # 注意：这里不使用torch.no_grad()，因为可能在训练模式
        feat = self.encoder.forward_features(window)["x_prenorm"][:, 1:]  # [B, 1369, 1024]
        
        return feat
    
    def accumulate_window_features(self, 
                                   feature_sum: torch.Tensor,
                                   count_map: torch.Tensor,
                                   window_feat: torch.Tensor,
                                   window_pos: Tuple[int, int, int, int]):
        """
        将窗口特征累加到全局特征图
        
        Args:
            feature_sum: 全局特征累加器 [B, H_patches, W_patches, C]
            count_map: 计数器 [H_patches, W_patches]
            window_feat: 窗口特征 [B, num_patches, C]
            window_pos: 窗口在原图的像素位置 (y1, y2, x1, x2)
        """
        y1, y2, x1, x2 = window_pos
        
        # 将像素位置转换为patch位置
        patch_y1 = y1 // self.patch_size
        patch_y2 = y2 // self.patch_size
        patch_x1 = x1 // self.patch_size
        patch_x2 = x2 // self.patch_size
        
        # 重塑窗口特征为2D grid [B, 37, 37, C]
        batch_size = window_feat.shape[0]
        window_feat_2d = window_feat.reshape(
            batch_size, self.window_patch_grid, self.window_patch_grid, -1
        )
        
        # 累加到全局特征图
        feature_sum[:, patch_y1:patch_y2, patch_x1:patch_x2, :] += window_feat_2d
        count_map[patch_y1:patch_y2, patch_x1:patch_x2] += 1
    
    def forward_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        使用滑动窗口提取高分辨率特征（串行处理）
        
        核心流程:
        1. 验证输入尺寸为1036x1036
        2. 串行处理9个518x518窗口
           - 逐个提取窗口特征
           - 累加到全局特征图
           - 节省显存
        3. 对重叠区域取平均
        4. 返回74x74的patch grid特征
        
        Args:
            img: 输入图像 [B, C, 1036, 1036]（已上采样）
            
        Returns:
            features: [B, 5476, 1024]
        """
        batch_size = img.shape[0]
        device = img.device
        feature_dim = self.encoder.embed_dim  # 1024 for DINOv2-large
        
        # 验证输入尺寸
        if img.shape[-2:] != (self.target_size, self.target_size):
            raise ValueError(
                f"Expected input size ({self.target_size}, {self.target_size}), "
                f"got {img.shape[-2:]}. Please upsample to 1036x1036 before calling this wrapper."
            )
        
        print(f"\n{'='*70}")
        print(f"Starting Sequential Feature Extraction")
        print(f"{'='*70}")
        print(f"Input shape:      {list(img.shape)}")
        print(f"Batch size:       {batch_size}")
        print(f"Device:           {device}")
        print(f"Processing:       {self.total_windows} windows sequentially...")
        print(f"{'='*70}\n")
        
        # 初始化特征累加器和计数器
        # feature_sum: 累加每个patch位置的特征向量
        # count_map: 记录每个patch被访问的次数（用于后续平均）
        feature_sum = torch.zeros(
            batch_size, self.final_patch_grid, self.final_patch_grid, feature_dim,
            device=device, dtype=torch.float32
        )
        count_map = torch.zeros(
            self.final_patch_grid, self.final_patch_grid,
            device=device, dtype=torch.float32
        )
        
        # 获取9个窗口位置
        window_positions = self.get_window_positions()
        
        # 串行处理每个窗口
        for window_idx, window_pos in enumerate(window_positions):
            y1, y2, x1, x2 = window_pos
            
            print(f"Window {window_idx + 1}/{self.total_windows}:")
            print(f"  Pixel region:  [{y1}:{y2}, {x1}:{x2}]")
            
            # 提取窗口特征 [B, 1369, 1024]
            window_feat = self.extract_window_features(img, window_pos)
            print(f"  Feature shape: {list(window_feat.shape)}")
            
            # 将像素位置转换为patch位置
            patch_y1 = y1 // self.patch_size
            patch_y2 = y2 // self.patch_size
            patch_x1 = x1 // self.patch_size
            patch_x2 = x2 // self.patch_size
            print(f"  Patch region:  [{patch_y1}:{patch_y2}, {patch_x1}:{patch_x2}]")
            
            # 累加到全局特征图
            self.accumulate_window_features(feature_sum, count_map, window_feat, window_pos)
            print(f"  ✓ Accumulated to global feature map\n")
        
        print(f"{'='*70}")
        print(f"All windows processed. Computing averaged features...")
        print(f"{'='*70}\n")
        
        # 对重叠区域取平均
        # count_map: [74, 74] → [1, 74, 74, 1]
        # feature_sum: [B, 74, 74, 1024]
        count_map_expanded = count_map.unsqueeze(0).unsqueeze(-1)  # [1, 74, 74, 1]
        feature_avg = feature_sum / (count_map_expanded + 1e-8)  # 避免除零
        
        # 重塑为 [B, num_patches, C]
        features = feature_avg.reshape(batch_size, -1, feature_dim)
        
        # 打印统计信息
        print(f"{'='*70}")
        print(f"Feature Extraction Completed!")
        print(f"{'='*70}")
        print(f"Output shape:     {list(features.shape)}")
        print(f"Expected:         [{batch_size}, {self.total_patches}, {feature_dim}]")
        print(f"\nOverlap Statistics:")
        print(f"  Min overlap:    {count_map.min().item():.0f}x")
        print(f"  Max overlap:    {count_map.max().item():.0f}x")
        print(f"  Mean overlap:   {count_map.mean().item():.2f}x")
        
        # 计算重叠区域分布
        unique_counts, patch_counts = torch.unique(count_map, return_counts=True)
        print(f"\nOverlap Distribution:")
        total_patches_check = 0
        for count, num_patches in zip(unique_counts, patch_counts):
            percentage = 100 * num_patches.item() / self.total_patches
            total_patches_check += num_patches.item()
            print(f"  {count.item():.0f}x overlap: {num_patches.item():4d} patches ({percentage:5.2f}%)")
        
        print(f"\nTotal patches:    {total_patches_check} (expected: {self.total_patches})")
        print(f"{'='*70}\n")
        
        return features
