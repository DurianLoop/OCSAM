import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import cv2
import ot
import math
from scipy.optimize import linear_sum_assignment

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# 导入DINOv2模块
from dinov2.models import vision_transformer as vits
from dinov2.models.vision_transformer import vit_small
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from matcher.k_means import kmeans_pp

import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import csv
import os 
from datetime import datetime

class MatchingMetrics:
    """评估匹配点准确性的指标类"""
    
    def __init__(self, patch_size=14, device='cuda'):
        self.patch_size = patch_size
        self.device = device
        
    def extract_true_points(self, true_mask, encoder_feat_size):
        # 池化得到patch级别的mask
        true_masks_pool = F.avg_pool2d(
            true_mask.float(), 
            (self.patch_size, self.patch_size)
        )
        
        # 使用阈值确定哪些patch属于目标
        true_masks_pool = (true_masks_pool > 0).float()
        
        # 获取所有值为1的patch的索引
        true_patches = torch.where(true_masks_pool.squeeze() == 1)
        
        # 转换为像素坐标
        true_points = []
        for h_idx, w_idx in zip(true_patches[0], true_patches[1]):
            # 计算patch中心的像素坐标
            x = w_idx * self.patch_size + self.patch_size // 2
            y = h_idx * self.patch_size + self.patch_size // 2
            true_points.append([x.item(), y.item()])
            
        return np.array(true_points)
    
    def compute_matching_accuracy(self, matched_points, true_points, distance_threshold=None):

        if distance_threshold is None:
            distance_threshold = 7
            
        metrics = {}
        
        # 处理边界情况
        if len(matched_points) == 0 or len(true_points) == 0:
            metrics['precision'] = 0.0 if len(matched_points) > 0 else 1.0
            metrics['coverage'] = 0.0 if len(true_points) > 0 else 1.0
            metrics['num_matched_points'] = len(matched_points)
            metrics['num_true_points'] = len(true_points)
            return metrics
        
        # 1. 计算距离矩阵
        dist_matrix = cdist(matched_points, true_points, metric='euclidean')
        
        # 2. 计算Precision：对每个matched point，检查是否有足够近的true point
        correct_matches = 0
        for m_idx in range(len(matched_points)):
            min_dist_to_true = dist_matrix[m_idx, :].min()
            if min_dist_to_true <= distance_threshold:
                correct_matches += 1
                
        precision = correct_matches / len(matched_points)
        
        # 3. 计算Coverage：true points中有多少被匹配点覆盖
        covered_true_points = 0
        for t_idx in range(len(true_points)):
            min_dist_from_matched = dist_matrix[:, t_idx].min()
            if min_dist_from_matched <= distance_threshold:
                covered_true_points += 1
                
        coverage = covered_true_points / len(true_points)
        
        metrics = {
            'precision': precision,
            'coverage': coverage,
            'num_matched_points': len(matched_points),
            'num_true_points': len(true_points),
            'num_correct_matches': correct_matches,
            'num_covered_true_points': covered_true_points,
            'distance_threshold': distance_threshold
        }
        
        return metrics


def evaluate_matching_accuracy(self, matched_points, true_mask):
    # 初始化评估器
    evaluator = MatchingMetrics(
        patch_size=self.encoder.patch_size,
        device=self.device
    )
    
    # 提取true points
    true_points = evaluator.extract_true_points(
        true_mask, 
        self.encoder_feat_size
    )
    
    # 计算指标
    metrics = evaluator.compute_matching_accuracy(
        matched_points, 
        true_points
    )
    
    # 打印结果
    print("\n=== Matching Accuracy Metrics ===")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['num_correct_matches']}/{metrics['num_matched_points']} matched points are correct)")
    print(f"Coverage: {metrics['coverage']:.3f} ({metrics['num_covered_true_points']}/{metrics['num_true_points']} true points are covered)")
    print("="*70)
    
    return metrics

class Matcher:
    def __init__(
            self,
            encoder,
            generator=None,
            input_size=518,
            num_centers=8,
            use_box=True,
            use_points_or_centers=True,
            sample_range=(4, 6),
            max_sample_iterations=30,
            alpha=1.,
            beta=0.,
            exp=0.,
            score_filter_cfg=True,
            num_merging_mask=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            target_color=None  # 修改：默认为 None
    ):
        # models
        self.encoder = encoder
        self.generator = generator
        self.rps = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None

        self.num_centers = num_centers
        self.use_box = True
        self.use_points_or_centers = False
        self.sample_range = sample_range
        self.max_sample_iterations = max_sample_iterations

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.device = device

        self.image_counter = 0

        # 修改：处理 target_color 为 None 的情况
        self.target_color = target_color if target_color else 'default'

    def _get_color_folder(self):
        """获取用于保存路径的颜色文件夹名"""
        if self.target_color and self.target_color != 'default':
            return f'_{self.target_color}'
        return ''

    def _get_mask_colormap(self):
        """获取 mask 显示的 colormap"""
        colormap_dict = {
            'red': 'Reds',
            'green': 'Greens', 
            'blue': 'Blues',
            'yellow': 'YlOrBr',
            'pink': 'RdPu',
            'default': 'Reds'  # MonuSeg 等数据集使用默认红色
        }
        return colormap_dict.get(self.target_color, 'Reds')

    def set_reference(self, imgs, masks):

        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        imgs = imgs.flatten(0, 1)
        img_size = imgs.shape[-1]
        assert img_size == self.input_size[-1]
        feat_size = img_size // self.encoder.patch_size

        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # process reference masks
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)

        # 保留像素级mask
        self.ref_masks_original = masks.clone()

        # 将像素级mask转变成patch级别的表示
        ref_masks_pool = F.avg_pool2d(masks, (self.encoder.patch_size, self.encoder.patch_size))
        nshot = ref_masks_pool.shape[0]

        # 保留池化之后的原始值作为权重
        self.ref_mask_weights = ref_masks_pool.clone().reshape(-1)
        
        ref_masks_pool = (ref_masks_pool > self.generator.predictor.model.mask_threshold).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)

        self.ref_imgs = imgs
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

    # Create grid
    def draw_patch_grid(self, ax, img_size=518, patch_size=14, color='black', alpha=0.3, linewidth=0.5):
        # 绘制垂直线
        for x in range(0, img_size + 1, patch_size):
            ax.axvline(x=x, color=color, alpha=alpha, linewidth=linewidth)
        
        # 绘制水平线
        for y in range(0, img_size + 1, patch_size):
            ax.axhline(y=y, color=color, alpha=alpha, linewidth=linewidth)

    def visualize_matching_results(self, ref_image, ref_mask, target_image, true_mask, points, true_points, save_path=None):
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        
        fig, axes = plt.subplots(1, 4, figsize=(30, 10))
        
        # === 子图1：Image with Input Mask ===
        ax = axes[0]
        
        if isinstance(ref_image, torch.Tensor):
            ref_img_display = ref_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if ref_img_display.max() <= 1.0:
                ref_img_display = (ref_img_display * 255).astype(np.uint8)
        else:
            ref_img_display = ref_image
        
        if isinstance(ref_mask, torch.Tensor):
            ref_mask_display = ref_mask.squeeze(0).squeeze(0).cpu().numpy()
        else:
            ref_mask_display = ref_mask

        # Create grid in figure 1
        self.draw_patch_grid(axes[0])
        
        ax.imshow(ref_img_display)
        
        # 使用辅助方法获取 colormap
        mask_color = self._get_mask_colormap()
        
        ax.imshow(ref_mask_display, alpha=0.5, cmap=mask_color)
        ax.set_title('Reference Image with Mask', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # === 子图2：Image with True Mask ===
        ax = axes[1]
        
        if isinstance(target_image, torch.Tensor):
            tar_img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if tar_img_display.max() <= 1.0:
                tar_img_display = (tar_img_display * 255).astype(np.uint8)
        else:
            tar_img_display = target_image
        
        ax.imshow(tar_img_display)
        
        if true_mask is not None:
            if isinstance(true_mask, torch.Tensor):
                true_mask_display = true_mask.squeeze(0).squeeze(0).cpu().numpy()
            else:
                true_mask_display = true_mask
            
            ax.imshow(true_mask_display, alpha=0.5, cmap=mask_color)
            ax.set_title('Target Image with Ground Truth Mask', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Target Image (No Ground Truth)', fontsize=14, fontweight='bold')
        
        ax.axis('off')
        
        # === 子图3：Image with Matched Points ===
        ax = axes[2]
        ax.imshow(tar_img_display)
        
        if points is not None and len(points) > 0:
            for i, (x, y) in enumerate(points):
                ax.scatter(x, y, s=30, c='red', marker='s', 
                          edgecolors='white', linewidth=0.5, zorder=3)
            
            ax.set_title(f'Matched Points on Target (Total: {len(points)})', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('No Matched Points Found', fontsize=14, fontweight='bold')
        
        ax.axis('off')

        # === 子图4：Image with True Points ===
        ax = axes[3]
        ax.imshow(tar_img_display)
        if true_points is not None and len(true_points) > 0:
            for i, (x, y) in enumerate(true_points):
                ax.scatter(x, y, s=30, c='green', marker='s', 
                          edgecolors='white', linewidth=0.5, zorder=3)
            ax.set_title(f'True Points from Ground Truth (Total: {len(true_points)})', 
                    fontsize=14, fontweight='bold')
        else:
            ax.set_title('No True Points Found', fontsize=14, fontweight='bold', color='gray')
    
        ax.axis('off')
        
        #标题中显示 target_color 信息
        title_suffix = f' - Target Color: {self.target_color}' if self.target_color != 'default' else ''
        fig.suptitle(f'Matching Results{title_suffix}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Combined visualization saved to {save_path}")
        
        plt.close()

    def visualize_rps_prompts(self, target_image, points, samples_list, save_path=None):
        """可视化RPS采样后的prompt points"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        n_groups = len(samples_list)
        total_plots = n_groups + 1
        cols = min(4, total_plots)
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        ax = axes[0]
        ax.imshow(img_display)
        
        if points is not None and len(points) > 0:
            is_cluster_centers = not self.use_points_or_centers

            if is_cluster_centers:
                for i, (x, y) in enumerate(points):
                    ax.scatter(x, y, marker='*', s=200, color='blue', 
                              edgecolors='white', linewidth=2, zorder=3)
                    ax.text(x+10, y-10, str(i), color='blue', 
                           fontsize=10, fontweight='bold')
                title_suffix = "(Cluster Centers)"
            else:
                for i, (x, y) in enumerate(points):
                    circle = patches.Circle((x, y), radius=6, color='blue', 
                                          fill=False, linewidth=2)
                    ax.add_patch(circle)
                    ax.text(x+8, y-8, str(i), color='blue', 
                           fontsize=10, fontweight='bold')
                title_suffix = "(All Points)"
                
            ax.set_title(f'Input Points {title_suffix}\n(Total: {len(points)})')
        else:
            ax.set_title('No Input Points')
        ax.axis('off')
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for group_idx, samples in enumerate(samples_list):
            if group_idx + 1 >= len(axes):
                break
                
            ax = axes[group_idx + 1]
            ax.imshow(img_display)
            
            if len(samples) > 0:
                representative_sample = samples[0]
                color = colors[group_idx % len(colors)]
                
                for point_idx, (x, y) in enumerate(representative_sample):
                    circle = patches.Circle((x, y), radius=6, color=color, fill=False, linewidth=2)
                    ax.add_patch(circle)
                    ax.text(x+8, y-8, str(point_idx), color=color, 
                           fontsize=10, fontweight='bold')
                
                points_per_sample = len(representative_sample)
                total_samples = len(samples)
                
                ax.set_title(f'Group {group_idx+1}: {points_per_sample} points\n'
                            f'(1 of {total_samples} samples shown)')
            else:
                ax.set_title(f'Group {group_idx+1}: Empty')
                
            ax.axis('off')
        
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"RPS prompts visualization saved to {save_path}")
        
        plt.close()

    def set_target(self, img, true_mask=None):
        
        img_h, img_w = img.shape[-2:]
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.tar_img = img
        self.tar_img_np = img_np

        if true_mask is not None:
            self.true_mask = true_mask
        else:
            self.true_mask = None
            print("Can not find the target true mask")

    def set_rps(self):
        if self.rps is None:
            assert self.encoder_feat_size is not None
            self.rps = RobustPromptSampler(
                encoder_feat_size=self.encoder_feat_size,
                sample_range=self.sample_range,
                max_iterations=self.max_sample_iterations
            )
    
    def predict(self):
        ref_feats, tar_feat = self.extract_img_feats()
        all_points, box, S, C, reduced_points_num = self.patch_level_matching(ref_feats=ref_feats, tar_feat=tar_feat)

        points = self.clustering(all_points) if not self.use_points_or_centers else all_points
        
        self.set_rps()

        pred_masks = self.mask_generation(self.tar_img_np, points, box, all_points, self.ref_masks_pool, C)
        
        return pred_masks

    def extract_img_feats(self):
        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
        tar_feat = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]
        print(f"After DINO, ref_feats shape is {ref_feats.shape}")
        
        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)
        tar_feat = tar_feat.reshape(-1, self.encoder.embed_dim)

        ref_feats = F.normalize(ref_feats, dim=1, p=2)
        tar_feat = F.normalize(tar_feat, dim=1, p=2)

        return ref_feats, tar_feat

    def patch_level_matching(self, ref_feats, tar_feat):
        #获取前景patch的布尔掩码
        mask_bool = self.ref_masks_pool.flatten().bool()

        ref_mask_feats = ref_feats[mask_bool]

        # Get weight in each patch
        weights = self.ref_mask_weights.flatten()[mask_bool]

        weights = weights.to(ref_mask_feats.device)

        # Normalize weights
        weights = weights / weights.sum()

        # weighted average
        ref_mask_feat_avg = (ref_mask_feats * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

        # Normalization
        ref_mask_feat_avg = F.normalize(ref_mask_feat_avg, dim=1, p=2)

        # 用平均特征与所有target特征计算相似度
        similarity_vector = ref_mask_feat_avg @ tar_feat.t()
        similarity_vector = similarity_vector.squeeze(0)

        # Using top-k to select patches
        k = 200
        topk_similarities, topk_indices = similarity_vector.topk(k=k)

        # MonuSeg threshold is 0.9
        threshold = 0.85
        selected_mask = topk_similarities > threshold
        points_matched_inds_set = topk_indices[selected_mask]

        points_matched_inds_set_w = points_matched_inds_set % (self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (self.encoder_feat_size)

        idxs_mask_set_x = (points_matched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        points_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_matched.append([int(x), int(y)])

        points = np.array(points_matched)

        color_folder = self._get_color_folder()
        if self.ref_imgs is not None and len(self.ref_imgs) > 0:
            save_path = f'debug_figure/matching_figure{color_folder}/combined_matching_{self.image_counter:04d}.png'
        
        # DINOv2 metric
        true_mask = self.true_mask if hasattr(self, 'true_mask') else None
        true_masks_pool = F.avg_pool2d(true_mask.float(), (self.encoder.patch_size, self.encoder.patch_size))
        
        true_points = []
        true_masks_pool = (true_masks_pool > self.generator.predictor.model.mask_threshold).float()

        true_patches = torch.where(true_masks_pool.squeeze() == 1)
        background_patches = torch.where(true_masks_pool.squeeze() == 0)

        W = self.encoder_feat_size
        fg_flat_indices = true_patches[0] * W + true_patches[1]
        bg_flat_indices = background_patches[0] * W + background_patches[1]

        forward_feat = tar_feat[fg_flat_indices]
        background_feat = tar_feat[bg_flat_indices]

        # Caculate average for forward features and background features
        forward_feat_avg = F.normalize(forward_feat.mean(dim=0, keepdim=True), dim=1, p=2)
        background_feat_avg = F.normalize(background_feat.mean(dim=0, keepdim=True), dim=1, p=2)

        # Avg Forward -- Background
        avg_fg_to_bg_sim = (forward_feat_avg @ background_feat.t()).mean()
        avg_fg_to_bg_dist = 1 - avg_fg_to_bg_sim

        print('='*70)
        print(f"Foreground-Background Separation (Avg FG → All BG):\n")
        print(f"  • Similarity: {avg_fg_to_bg_sim:.4f} ↓ (target: <0.3)")
        print(f"  • Distance:   {avg_fg_to_bg_dist:.4f} ↑ (target: >0.7)")
        print('='*70)
        
        for h_idx, w_idx in zip(true_patches[0], true_patches[1]):
            x = w_idx * self.encoder.patch_size + self.encoder.patch_size // 2
            y = h_idx * self.encoder.patch_size + self.encoder.patch_size // 2
            true_points.append([x.item(), y.item()])
            
        true_points = np.array(true_points) if true_points else np.array([])
        
        self.visualize_matching_results(
            ref_image=self.ref_imgs[0],
            ref_mask=self.ref_masks_original[0],
            target_image=self.tar_img,
            true_mask=true_mask,
            points=points,
            true_points=true_points,
            save_path=save_path
        )

        metrics = evaluate_matching_accuracy(self, points, true_mask)
        
        if self.use_box and points.shape[0] > 0:
            box = np.array([
                max(points[:, 0].min(), 0),
                max(points[:, 1].min(), 0),
                min(points[:, 0].max(), self.input_size[1] - 1),
                min(points[:, 1].max(), self.input_size[0] - 1),
            ])
        else:
            box = None
            print('No points found')

        S = ref_feats @ tar_feat.t()
        C = (1 - S) / 2
    
        reduced_points_num = len(points)

        return points, box, S, C, reduced_points_num

    def clustering(self, points):

        num_centers = min(self.num_centers, len(points))
        flag = True
        while (flag):
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            id, fre = torch.unique(cluster_assignment, return_counts=True)
            if id.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers

    def visualize_sam_outputs(self, target_image, tar_masks_ori, tar_masks, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import random
        
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        total_masks = len(tar_masks_ori)
        
        if total_masks == 0:
            print("No masks to visualize")
            return
        
        num_to_show = min(16, total_masks)
        if total_masks > 16:
            selected_indices = random.sample(range(total_masks), 16)
        else:
            selected_indices = list(range(total_masks))
        
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()
        
        # 使用辅助方法获取 colormap
        mask_color = self._get_mask_colormap()
        
        for i in range(16):
            ax = axes[i]
            
            if i < len(selected_indices):
                mask_idx = selected_indices[i]
                
                ax.imshow(img_display)
                
                mask = tar_masks[mask_idx].squeeze()
                if mask.sum() > 0:
                    ax.imshow(mask, alpha=0.5, cmap=mask_color)
                
                mask_info = tar_masks_ori[mask_idx]
                area = mask_info.get('area', 0)
                iou = mask_info.get('predicted_iou', 0)
                
                ax.set_title(f'Mask {mask_idx+1}\nArea: {area:.0f}\nIoU: {iou:.3f}', 
                            fontsize=10)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No Mask', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            ax.axis('off')
        
        fig.suptitle(f'SAM Output Masks Preview ({num_to_show}/{total_masks} shown)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"SAM masks visualization saved to {save_path}")
        
        plt.close()

    
    def mask_generation(self, tar_img_np, points, box, all_points, ref_masks_pool, C):
        
        samples_list, label_list = self.rps.sample_points(points)
        
        color_folder = self._get_color_folder()
        rps_save_path = f'debug_figure/matching_figure{color_folder}/rps_prompts_{self.image_counter:04d}.png'
        self.visualize_rps_prompts(self.tar_img, points, samples_list, save_path=rps_save_path)

        tar_masks_ori = self.generator.generate(
            tar_img_np,
            select_point_coords=samples_list,
            select_point_labels=label_list,
            select_box=[box] if self.use_box else None,
        )

        tar_masks = torch.cat(
            [torch.from_numpy(qmask['segmentation']).float()[None, None, ...].to(self.device) for
             qmask in tar_masks_ori], dim=0).cpu().numpy() > 0

        # 修改：使用辅助方法构建保存路径
        sam_masks_save_path = f'debug_figure/sam_figure{color_folder}/sam_outputs_{self.image_counter:04d}.png'
        self.visualize_sam_outputs(self.tar_img, tar_masks_ori, tar_masks, save_path=sam_masks_save_path)

        purity = torch.zeros(tar_masks.shape[0])
        coverage = torch.zeros(tar_masks.shape[0])
        emd = torch.zeros(tar_masks.shape[0])

        samples = samples_list[-1]
        labels = torch.ones(tar_masks.shape[0], samples.shape[1])
        samples = torch.ones(tar_masks.shape[0], samples.shape[1], 2)

        for i in range(len(tar_masks)):
            purity_, coverage_, emd_, sample_, label_, mask_ = \
                self.rps.get_mask_scores(
                    points=points,
                    masks=tar_masks[i],
                    all_points=all_points,
                    emd_cost=C,
                    ref_masks_pool=ref_masks_pool
                )
            assert np.all(mask_ == tar_masks[i])
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        pred_masks = tar_masks.squeeze(1)
        metric_preds = {
            "purity": purity,
            "coverage": coverage,
            "emd": emd
        }

        scores = self.alpha * emd + self.beta * purity * coverage ** self.exp

        def check_pred_mask(pred_masks):
            if len(pred_masks.shape) < 3:
                pred_masks = pred_masks[None, ...]
            return pred_masks

        pred_masks = check_pred_mask(pred_masks)

        for metric in ["coverage", "emd", "purity"]:
            if self.score_filter_cfg[metric] > 0:
                thres = min(self.score_filter_cfg[metric], metric_preds[metric].max())
                idx = torch.where(metric_preds[metric] >= thres)[0]
                scores = scores[idx]
                samples = samples[idx]
                labels = labels[idx]
                pred_masks = check_pred_mask(pred_masks[idx])

                for key in metric_preds.keys():
                    metric_preds[key] = metric_preds[key][idx]

        if self.score_filter_cfg["score_filter"]:

            distances = 1 - scores
            distances, rank = torch.sort(distances, descending=False)

            distances_norm = distances - distances.min()
            distances_norm = distances_norm / (distances.max() + 1e-6)
            filer_dis = distances < self.score_filter_cfg["score"]
            filer_dis[..., 0] = True
            filer_dis_norm = distances_norm < self.score_filter_cfg["score_norm"]
            filer_dis = filer_dis * filer_dis_norm

            pred_masks = check_pred_mask(pred_masks)

            masks = pred_masks[rank[filer_dis][:self.num_merging_mask]]
            masks = check_pred_mask(masks)
            
            masks = masks.sum(0) > 0
            masks = masks[None, ...]

        else:
            topk = min(self.num_merging_mask, scores.size(0))
            topk_idx = scores.topk(topk)[1]
            topk_samples = samples[topk_idx].cpu().numpy()
            topk_scores = scores[topk_idx].cpu().numpy()
            topk_pred_masks = pred_masks[topk_idx]
            topk_pred_masks = check_pred_mask(topk_pred_masks)

            if self.score_filter_cfg["topk_scores_threshold"] > 0:
                topk_scores = topk_scores / (topk_scores.max())

            idx = topk_scores > self.score_filter_cfg["topk_scores_threshold"]
            topk_samples = topk_samples[idx]

            topk_pred_masks = check_pred_mask(topk_pred_masks)
            topk_pred_masks = topk_pred_masks[idx]
            mask_list = []
            for i in range(len(topk_samples)):
                mask = topk_pred_masks[i][None, ...]
                mask_list.append(mask)
            masks = np.sum(mask_list, axis=0) > 0
            masks = check_pred_mask(masks)

        self.image_counter += 1

        return torch.tensor(masks, device=self.device, dtype=torch.float)

    def clear(self):

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None

        self.encoder_img_size = None
        self.encoder_feat_size = None


class RobustPromptSampler:

    def __init__(
        self,
        encoder_feat_size,
        sample_range,
        max_iterations
    ):
        self.encoder_feat_size = encoder_feat_size
        self.sample_range = sample_range
        self.max_iterations = max_iterations

    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):

        def is_in_mask(point, mask):
            h, w = mask.shape
            point = point.astype(np.int)
            point = point[:, ::-1]
            point = np.clip(point, 0, [h - 1, w - 1])
            return mask[point[:, 0], point[:, 1]]

        ori_masks = masks
        masks = cv2.resize(
            masks[0].astype(np.float32),
            (self.encoder_feat_size, self.encoder_feat_size),
            interpolation=cv2.INTER_AREA)
        if masks.max() <= 0:
            thres = masks.max() - 1e-6
        else:
            thres = 0
        masks = masks > thres

        emd_cost_pool = emd_cost[ref_masks_pool.flatten().bool(), :][:, masks.flatten()]
        emd = ot.emd2(a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
                      b=[1. / emd_cost_pool.shape[1] for i in range(emd_cost_pool.shape[1])],
                      M=emd_cost_pool.cpu().numpy())
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]

        mask_area = max(float(masks.sum()), 1.0)
        purity = points_in_mask.shape[0] / mask_area
        coverage = points_in_mask.shape[0] / all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, points, labels, ori_masks

    def combinations(self, n, k):
        if k > n:
            return []
        if k == 0:
            return [[]]
        if k == n:
            return [[i for i in range(n)]]
        res = []
        for i in range(n):
            for j in self.combinations(i, k - 1):
                res.append(j + [i])
        return res

    def sample_points(self, points):
        
        sample_list = []
        label_list = []
        
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            if len(points) > 8:
                index = [random.sample(range(len(points)), i) for j in range(self.max_iterations)]
                sample = np.take(points, index, axis=0)
                
            else:
                index = self.combinations(len(points), i)
                sample = np.take(points, index, axis=0)

            label = np.ones((sample.shape[0], i))
            sample_list.append(sample)
            label_list.append(label)
            
        return sample_list, label_list


def build_matcher_oss(args):

    # DINOv2, Image Encoder
    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )

    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)
    print(f"Successfully loading {args.dinov2_size}")

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    dinov2.to(device=args.device)

    # SAM
    sam = sam_model_registry[args.sam_size](checkpoint=args.sam_weights)
    sam.to(device=args.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=1.0,
        sel_stability_score_thresh=args.sel_stability_score_thresh,
        sel_pred_iou_thresh=args.iou_filter,
        box_nms_thresh=args.box_nms_thresh,
        sel_output_layer=args.output_layer,
        output_layer=args.dense_multimask_output,
        dense_pred=args.use_dense_mask,
        multimask_output=args.dense_multimask_output > 0,
        sel_multimask_output=args.multimask_output > 0,
    )

    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": True,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }
    
    target_color = getattr(args, 'target_color', None)

    return Matcher(
        encoder=dinov2,
        generator=generator,
        use_box=args.use_box,
        use_points_or_centers=args.use_points_or_centers,
        sample_range=args.sample_range,
        max_sample_iterations=args.max_sample_iterations,
        alpha=args.alpha,
        beta=args.beta,
        exp=args.exp,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=args.num_merging_mask,
        device=args.device,
        target_color=target_color
    )