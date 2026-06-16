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
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from dinov2_wrapper import DINOv2SlidingWindowWrapper

from matcher.k_means import kmeans_pp

import random

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
            target_color='blue',

            # Unsampling
            target_size=1036,
            wrapper=None,
            use_sliding_window=True
    ):
        # models
        self.encoder = encoder
        self.wrapper = wrapper
        self.generator = generator
        self.rps = None

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # Upsampling
        self.target_size=target_size
        self.use_sliding_window = use_sliding_window

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(), #将图像转换成PyTorch tensor格式
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 进行归一化
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
        self.use_points_or_centers = True
        self.sample_range = sample_range
        self.max_sample_iterations =max_sample_iterations

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.device = device

        self.image_counter=0

        self.target_color = target_color
        print(f'Matcher ensure target color is {target_color}')

        if use_sliding_window:
            self.encoder_img_size = target_size  # 1036
            self.encoder_feat_size = target_size // encoder.patch_size  # 74
            print(f"\n{'='*70}")
            print(f"✓ Using Sliding Window Feature Extraction")
            print(f"  Target size: {target_size}x{target_size}")
            print(f"  Feature map: {self.encoder_feat_size}x{self.encoder_feat_size} ({self.encoder_feat_size**2} patches)")
            print(f"{'='*70}\n")
        else:
            self.encoder_img_size = input_size[0]  # 518
            self.encoder_feat_size = input_size[0] // encoder.patch_size  # 37
            print(f"\n{'='*70}")
            print(f"✓ Using Standard Feature Extraction")
            print(f"  Feature map: {self.encoder_feat_size}x{self.encoder_feat_size} ({self.encoder_feat_size**2} patches)")
            print(f"{'='*70}\n")

    def set_reference(self, imgs, masks):

        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        imgs = imgs.flatten(0, 1)  # bs, 3, h, w
        img_size = imgs.shape[-1]
        assert img_size == self.input_size[-1]
        
        # feat_size = img_size // self.encoder.patch_size
        feat_size = img_size // self.encoder.patch_size
        # # print('Patch Size is',self.encoder.patch_size)
        
        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # process reference masks
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)  # ns, 1, h, w
        nshot = masks.shape[0]

        # 保留像素级mask
        self.ref_masks_original=masks.clone()
        print("Reference Mask is:",masks)

        if self.use_sliding_window:
            masks_upsampled = F.interpolate(
                    masks.float(),
                    size=(1036, 1036),
                    mode='bicubic',
                    align_corners = False
            )
            
            ref_masks_pool = F.avg_pool2d(
                    masks_upsampled,
                    (self.encoder.patch_size,self.encoder.patch_size)
            )
        else:
            ref_masks_pool = F.avg_pool2d(
                masks,
                (self.encoder.patch_size, self.encoder.patch_size)
            )
        
        # 将像素级mask转变成patch级别的表示
        # My idea: 这个只针对于大面积的mask作用明显, 对于一些比较小的mask效果并不会很好
        # ref_masks_pool = F.avg_pool2d(masks, (self.encoder.patch_size, self.encoder.patch_size))
        
        nshot = ref_masks_pool.shape[0]
        ref_masks_pool = (ref_masks_pool > self.generator.predictor.model.mask_threshold).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)  # nshot, N
        
        self.ref_imgs = imgs
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

    def visualize_matching_results(self, ref_image, ref_mask, target_image, true_mask, points, save_path=None):
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        
        # 创建1行3列的子图
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # === 子图1：Reference Image with Mask ===
        ax = axes[0]
        
        # 转换reference图像格式
        if isinstance(ref_image, torch.Tensor):
            ref_img_display = ref_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if ref_img_display.max() <= 1.0:
                ref_img_display = (ref_img_display * 255).astype(np.uint8)
        else:
            ref_img_display = ref_image
        
        # 转换reference mask格式
        if isinstance(ref_mask, torch.Tensor):
            ref_mask_display = ref_mask.squeeze(0).squeeze(0).cpu().numpy()
        else:
            ref_mask_display = ref_mask
        
        # 显示reference图像和mask
        ax.imshow(ref_img_display)
        
        # 根据target_color选择颜色
        colormap_dict = {
            'red': 'Reds',
            'green': 'Greens', 
            'blue': 'Blues',
            'yellow': 'YlOrBr',
            'pink': 'RdPu'
        }
        mask_color = colormap_dict.get(self.target_color, 'Reds')
        
        ax.imshow(ref_mask_display, alpha=0.5, cmap=mask_color)
        ax.set_title('Reference Image with Mask', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # === 子图2：Target Image with True Mask ===
        ax = axes[1]
        
        # 转换target图像格式
        if isinstance(target_image, torch.Tensor):
            tar_img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if tar_img_display.max() <= 1.0:
                tar_img_display = (tar_img_display * 255).astype(np.uint8)
        else:
            tar_img_display = target_image
        
        # 显示target图像
        ax.imshow(tar_img_display)
        
        # 如果有true mask，显示它
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
        
        # === 子图3：Target Image with Matched Points ===
        ax = axes[2]
        
        # 显示target图像
        ax.imshow(tar_img_display)
        
        # 在图像上标记匹配的points
        if points is not None and len(points) > 0:
            # 使用实心小点标记
            for i, (x, y) in enumerate(points):
                # 绘制实心点
                ax.scatter(x, y, s=30, c='red', marker='o', 
                          edgecolors='white', linewidth=0.5, zorder=3)
                
                # 添加编号标签（可选，如果点太多可以注释掉）
                if len(points) <= 50:  # 只在点数不太多时显示标签
                    ax.text(x+5, y-5, str(i), color='yellow', 
                           fontsize=8, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='black', alpha=0.5))
            
            ax.set_title(f'Matched Points on Target (Total: {len(points)})', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('No Matched Points Found', fontsize=14, fontweight='bold')
        
        ax.axis('off')
        
        # 添加总标题
        fig.suptitle(f'Matching Results - Target Color: {self.target_color}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 调整子图间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Combined visualization saved to {save_path}")
        
        plt.close()
        
        # 打印统计信息
        print(f"Reference mask pixels: {ref_mask_display.sum() if ref_mask is not None else 0}")
        if true_mask is not None:
            print(f"Target true mask pixels: {true_mask_display.sum()}")
        print(f"Number of matched points: {len(points) if points is not None else 0}")

    def visualize_rps_prompts(self, target_image, points, samples_list, save_path=None):
        """
        可视化RPS采样后的prompt points - 每个group只显示一个采样
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        
        # 转换图像格式
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        # 创建子图：原始points + 每个group的一个采样
        n_groups = len(samples_list)
        total_plots = n_groups + 1  # +1 for input points
        cols = min(4, total_plots)
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        # 处理axes格式
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        # 第一个子图：显示输入的points
        ax = axes[0]
        ax.imshow(img_display)
        
        if points is not None and len(points) > 0:
            # 判断points来源
            is_cluster_centers= not self.use_points_or_centers

            if is_cluster_centers:
                for i, (x, y) in enumerate(points):
                    ax.scatter(x, y, marker='*', s=200, color='blue', 
                              edgecolors='white', linewidth=2, zorder=3)
                    ax.text(x+10, y-10, str(i), color='blue', 
                           fontsize=10, fontweight='bold')
                title_suffix = "(Cluster Centers)"
            
            else:
                # 使用圆点标记all points
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
        
        # 为每个group显示一个代表性采样
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for group_idx, samples in enumerate(samples_list):
            if group_idx + 1 >= len(axes):
                break
                
            ax = axes[group_idx + 1]
            ax.imshow(img_display)
            
            if len(samples) > 0:
                # 选择第一个采样作为代表
                representative_sample = samples[0]
                color = colors[group_idx % len(colors)]
                
                # 显示这个采样中的所有点
                for point_idx, (x, y) in enumerate(representative_sample):
                    circle = patches.Circle((x, y), radius=6, color=color, fill=False, linewidth=2)
                    ax.add_patch(circle)
                    ax.text(x+8, y-8, str(point_idx), color=color, 
                           fontsize=10, fontweight='bold')
                
                # 根据sample_range推断点数
                points_per_sample = len(representative_sample)
                total_samples = len(samples)
                
                ax.set_title(f'Group {group_idx+1}: {points_per_sample} points\n'
                            f'(1 of {total_samples} samples shown)')
            else:
                ax.set_title(f'Group {group_idx+1}: Empty')
                
            ax.axis('off')
        
        # 隐藏多余的子图
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
        print("The shape of image height",img_h)
        print("The shape of image weight",img_w)
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        # transform query to numpy as input of sam
        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.tar_img = img
        self.tar_img_np = img_np

        if true_mask is not None:
            self.true_mask=true_mask
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

        # Robust Prompt Sampler
        # 选择points (方法二选一)
        points = self.clustering(all_points) if not self.use_points_or_centers else all_points
        
        self.set_rps()

        pred_masks = self.mask_generation(self.tar_img_np, points, box, all_points, self.ref_masks_pool, C)
        return pred_masks

    # 使用DINOv2进行特征提取
    def extract_img_feats(self):
        
        #处理reference and target image
        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        print("reference image shape is",ref_imgs.shape)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        if self.use_sliding_window and self.wrapper is not None:
            # Upsmaling
            ref_imgs = F.interpolate(ref_imgs, size=(1036, 1036), mode='bicubic', align_corners=False)
            tar_img = F.interpolate(tar_img, size=(1036, 1036), mode='bicubic', align_corners=False)
            print(f"Reference image shape {ref_imgs.shape} and Target image shape{tar_img.shape} after upsampling")
            
            # Extract Feature
            with torch.no_grad():
                ref_feats = self.wrapper.forward_features(ref_imgs.to(self.device))
                print("ref_feats shape",ref_feats.shape)
                tar_feat = self.wrapper.forward_features(tar_img.to(self.device))
                print("ref_feats shape",ref_feats.shape)
        else:
            ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
            tar_feat = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]

        # ns, N, c = ref_feats.shape
        # 目的是去除batch
        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)  # ns*N, c
        tar_feat = tar_feat.reshape(-1, self.encoder.embed_dim)  # N, c

        # Normalization
        ref_feats = F.normalize(ref_feats, dim=1, p=2) # normalize for cosine similarity
        tar_feat = F.normalize(tar_feat, dim=1, p=2)

        return ref_feats, tar_feat

    # Matching
    def patch_level_matching(self, ref_feats, tar_feat):

        ref_mask_feats=ref_feats[self.ref_masks_pool.flatten().bool()]
        print(f"Reference mask patches: {ref_mask_feats.shape[0]}")

        similarity_matrix = ref_mask_feats @ tar_feat.t()

        topk_similarities, topk_indices = similarity_matrix.topk(k=10, dim=1)

        threshold = 0.8
        selected_similarity_mask = topk_similarities > threshold
        matched_target_indices = topk_indices[selected_similarity_mask]

        # 去重
        points_matched_inds_set = torch.unique(matched_target_indices)    

        ###################### Threshold ###################################################

        max_similarity_tar = similarity_matrix.max(dim=0).values

        # threshold = 0.8

        # # 选择相似度超过阈值的target patches
        # selected_similarity_mask = max_similarity_tar > threshold
        # matched_target_indices = torch.where(selected_similarity_mask)[0]

        # #去重
        # points_matched_inds_set = torch.unique(matched_target_indices)    
        
        # 将patch索引转换成像素坐标点作为prompt中的Point
        # 将一维索引转成二维网格坐标
        points_matched_inds_set_w = points_matched_inds_set % (self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (self.encoder_feat_size)

        # 得到像素坐标
        idxs_mask_set_x = (points_matched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        points_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_matched.append([int(x), int(y)])

        #转换成numpy数组
        #里面装有有效(x,y)像素坐标点
        points = np.array(points_matched)
        print('shape of points as prompt',points.shape)

        # Visualize Matching Information
        if self.ref_imgs is not None and len(self.ref_imgs) >0:
            save_path = f'debug_figure/matching_figure_{self.target_color}/combined_matching_{self.image_counter:04d}.png'
        
        # 获取true_mask
        true_mask = self.true_mask if hasattr(self, 'true_mask') else None

        self.visualize_matching_results(
            ref_image=self.ref_imgs[0],
            ref_mask=self.ref_masks_original[0],
            target_image=self.tar_img,
            true_mask=true_mask,
            points=points,
            save_path=save_path
        )

        # Add box prompt
        # print(f'self.use_box: {self.use_box}')
        # print(f'points.shape: {points.shape}')
        # print(f'points count: {points.shape[0] if len(points.shape) > 0 else 0}')
        
        if self.use_box and points.shape[0]>0:
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

    #使用K-means将匹配点聚类为num_centers(8)个簇
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


    # Visualize mask after sam
    def visualize_sam_outputs(self, target_image, tar_masks_ori, tar_masks, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import random
        
        # 转换图像格式
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        # 获取masks数量
        total_masks = len(tar_masks_ori)
        print(f"Total SAM output masks: {total_masks}")
        
        if total_masks == 0:
            print("No masks to visualize")
            return
        
        # 随机选择最多16个mask进行展示
        num_to_show = min(16, total_masks)
        if total_masks > 16:
            selected_indices = random.sample(range(total_masks), 16)
        else:
            selected_indices = list(range(total_masks))
        
        # 创建3x3子图
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()
        
        # 根据target_color选择颜色
        colormap_dict = {
            'red': 'Reds',
            'green': 'Greens', 
            'blue': 'Blues',
            'yellow': 'YlOrBr',
            'pink': 'RdPu'
        }
        mask_color = colormap_dict.get(self.target_color, 'Reds')
        
        for i in range(16):
            ax = axes[i]
            
            if i < len(selected_indices):
                mask_idx = selected_indices[i]
                
                # 显示原图
                ax.imshow(img_display)
                
                # 叠加mask
                mask = tar_masks[mask_idx].squeeze()
                if mask.sum() > 0:  # 只有当mask不为空时才显示
                    ax.imshow(mask, alpha=0.5, cmap=mask_color)
                
                # 获取mask信息
                mask_info = tar_masks_ori[mask_idx]
                area = mask_info.get('area', 0)
                iou = mask_info.get('predicted_iou', 0)
                
                ax.set_title(f'Mask {mask_idx+1}\nArea: {area:.0f}\nIoU: {iou:.3f}', 
                            fontsize=10)
            else:
                # 空白子图
                ax.axis('off')
                ax.text(0.5, 0.5, 'No Mask', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            ax.axis('off')
        
        # 添加总标题
        fig.suptitle(f'SAM Output Masks Preview ({num_to_show}/{total_masks} shown)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"SAM masks visualization saved to {save_path}")
        
        plt.close()

    
    def mask_generation(self, tar_img_np, points, box, all_points, ref_masks_pool, C):
        # RPS sampling
        samples_list, label_list = self.rps.sample_points(points)

        # Visualize RPS result
        rps_save_path = f'debug_figure/matching_figure_{self.target_color}/rps_prompts_{self.image_counter:04d}.png'
        self.visualize_rps_prompts(self.tar_img, points, samples_list, save_path=rps_save_path)
        
        # 打印采样信息
        print(f"RPS Input: {'All Points' if self.use_points_or_centers else 'Clustering Centers'}")
        print(f"Input points count: {len(points) if points is not None else 0}")
        for i, samples in enumerate(samples_list):
            print(f"Group {i+1}: {len(samples[0])} points per sample, {len(samples)} total samples")

        tar_masks_ori = self.generator.generate(
            tar_img_np,
            # 点提示
            select_point_coords=samples_list,
            # 点标签
            select_point_labels=label_list,
            select_box=[box] if self.use_box else None,
        )

        # tar_masks目的是将sam原始输出转换成统一的tensor格式
        # tar_masks是预测出来的target masks
        tar_masks = torch.cat(
            [torch.from_numpy(qmask['segmentation']).float()[None, None, ...].to(self.device) for
             qmask in tar_masks_ori], dim=0).cpu().numpy() > 0

        sam_masks_save_path = f'debug_figure/sam_figure_{self.target_color}/sam_outputs_{self.image_counter:04d}.png'
        self.visualize_sam_outputs(self.tar_img, tar_masks_ori, tar_masks, save_path=sam_masks_save_path)


        # append to original results
        purity = torch.zeros(tar_masks.shape[0])
        coverage = torch.zeros(tar_masks.shape[0])
        emd = torch.zeros(tar_masks.shape[0])

        samples = samples_list[-1]
        labels = torch.ones(tar_masks.shape[0], samples.shape[1])
        samples = torch.ones(tar_masks.shape[0], samples.shape[1], 2)

        # compute scores for each mask
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

        #确定shape完整性
        def check_pred_mask(pred_masks):
            if len(pred_masks.shape) < 3:  # avoid only one mask
                pred_masks = pred_masks[None, ...]
            return pred_masks

        pred_masks = check_pred_mask(pred_masks)

        # filter the false-positive mask fragments by using the proposed metrics
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

        #  score-based masks selection, masks merging
        # 距离筛选方法
        if self.score_filter_cfg["score_filter"]:

            distances = 1 - scores
            distances, rank = torch.sort(distances, descending=False)

            # Normalization -- 平移
            distances_norm = distances - distances.min()
            # Normalization -- 缩放
            distances_norm = distances_norm / (distances.max() + 1e-6)
            filer_dis = distances < self.score_filter_cfg["score"]
            filer_dis[..., 0] = True
            filer_dis_norm = distances_norm < self.score_filter_cfg["score_norm"]
            filer_dis = filer_dis * filer_dis_norm

            pred_masks = check_pred_mask(pred_masks)
            print("Shape of pred_masks:",pred_masks.shape)

            #对所有masks进行筛选并且排序(选择10个)
            masks = pred_masks[rank[filer_dis][:self.num_merging_mask]]
            masks = check_pred_mask(masks)
            print("Before merge, shape of masks is (has Top10 masks)",masks.shape)
            
            masks = masks.sum(0) > 0
            masks = masks[None, ...]

        # 直接选择分数最高的K个掩码
        else:
            # 选择10个掩码最高的
            topk = min(self.num_merging_mask, scores.size(0))
            topk_idx = scores.topk(topk)[1]
            topk_samples = samples[topk_idx].cpu().numpy()
            topk_scores = scores[topk_idx].cpu().numpy()
            topk_pred_masks = pred_masks[topk_idx]
            topk_pred_masks = check_pred_mask(topk_pred_masks)

            if self.score_filter_cfg["topk_scores_threshold"] > 0:
                # map scores to 0-1
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


    # masks进行确认(target mask ground truth进行泄露)
    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):

        def is_in_mask(point, mask):
            # input: point: n*2, mask: h*w
            # output: n*1
            h, w = mask.shape
            point = point.astype(np.int)
            point = point[:, ::-1]  # y,x
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

        # 1. emd
        emd_cost_pool = emd_cost[ref_masks_pool.flatten().bool(), :][:, masks.flatten()]
        emd = ot.emd2(a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
                      b=[1. / emd_cost_pool.shape[1] for i in range(emd_cost_pool.shape[1])],
                      M=emd_cost_pool.cpu().numpy())
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        # 2. purity and coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]
        # here we define two metrics for local matching , purity and coverage
        # purity: points_in/mask_area, the higher means the denser points in mask
        # coverage: points_in / all_points, the higher means the mask is more complete
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

    # 选择points作为prompt
    def sample_points(self, points):
        
        # return list of arrary
        sample_list = []
        label_list = []
        
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            if len(points) > 8:
                index = [random.sample(range(len(points)), i) for j in range(self.max_iterations)]
                sample = np.take(points, index, axis=0)  # (max_iterations * i) * 2
                
            else:
                index = self.combinations(len(points), i)
                sample = np.take(points, index, axis=0)  # i * n * 2

            # generate label  max_iterations * i
            # label指的是SAM点提示的标签, 正常情况下label=1表示前景,label=0表示背景
            #在这里所有sample的label都设置为1
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

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    dinov2.to(device=args.device)
    
    # Wrapper
    use_sliding_window = getattr(args, 'use_sliding_window', True)
    if use_sliding_window:
        print("\nCreating sliding window wrapper...")
        wrapper = DINOv2SlidingWindowWrapper(
            encoder=dinov2,
            patch_size=14,
            window_size=518,
            target_size=1036
        )
        print("✓ Wrapper created")
    else:
        wrapper = None
        print("✓ Using standard extraction (no wrapper)")


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
        # "score_filter": args.use_score_filter,
        "score_filter": True,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    return Matcher(
        encoder=dinov2,
        wrapper=wrapper,
        generator=generator,
        num_centers=args.num_centers,
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
        target_color=args.target_color,
    )
