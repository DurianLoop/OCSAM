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
from segment_anything import SamPredictor

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


class Matcher:
    def __init__(
            self,
            encoder,
            predictor=None,
            input_size=518,
            num_centers=10,
            box_size=15,
            use_box=True,
            sample_range=(4, 6),
            max_sample_iterations=30,
            alpha=1.,
            beta=0.,
            exp=0.,
            score_filter_cfg=True,
            num_merging_mask=10,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            target_color='blue'
    ):
        
        # models
        self.encoder = encoder
        self.predictor = predictor

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
        self.box_size = box_size  # 每个center周围的box大小（像素）
        self.use_box = use_box

        self.alpha, self.beta, self.exp = alpha, beta, exp
        assert score_filter_cfg is not None
        self.score_filter_cfg = score_filter_cfg
        self.num_merging_mask = num_merging_mask

        self.device = device
        self.image_counter = 0
        self.target_color = target_color

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
        
        threshold = 0
        ref_masks_pool = (ref_masks_pool > threshold).float()
        ref_masks_pool = ref_masks_pool.reshape(-1)

        self.ref_imgs = imgs
        self.ref_masks_pool = ref_masks_pool
        self.nshot = nshot

    def set_target(self, img, true_mask=None):
        
        img_h, img_w = img.shape[-2:]
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        # transform query to numpy as input of sam
        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.tar_img = img
        self.tar_img_np = img_np

        if true_mask is not None:
            self.true_mask = true_mask
        else:
            self.true_mask = None

    def predict(self):
        # 1. 提取特征
        ref_feats, tar_feat = self.extract_img_feats()
        
        # 2. Patch级别匹配，获取所有匹配点
        all_points = self.patch_level_matching(ref_feats=ref_feats, tar_feat=tar_feat)

        # 将matched points 写入csv folder
        self.save_points_to_csv(all_points)
        
        if len(all_points) == 0:
            print("No matching points found!")
            return torch.zeros(1, self.input_size[0], self.input_size[1], device=self.device)
        
        # 3. 聚类得到 centers (固定使用聚类中心)
        # centers = self.clustering(all_points)
        # print(f"Clustering centers: {len(centers)} centers")
        
        # 4. 为每个 center 创建小的 bounding box
        center_boxes = self.create_center_boxes(all_points)
        print(f"Created {len(center_boxes)} boxes around centers")

        # # if we use all points to create boudning box
        # center_boxes = self.create_center_boxes(all_points)
        # print(f"Created {len(center_boxes)} boxes around centers")
        
        # 5. 使用 box-only prompt 进行 mask generation
        pred_masks = self.mask_generation(self.tar_img_np, center_boxes)
        
        return pred_masks

    def save_points_to_csv(self, points):
        # 创建保存目录
        save_dir = f'IDRID/{self.target_color}/Match_Points_{self.target_color}/'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        csv_filename = f'matched_points_{self.image_counter:04d}.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        
        # 写入 CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['point_id', 'x', 'y'])
            # 写入数据
            for idx, point in enumerate(points):
                writer.writerow([idx, point[0], point[1]])
        
        print(f"Saved {len(points)} points to {csv_path}")

    def extract_img_feats(self):
        # 处理reference and target image
        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        # Extract Feature
        ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
        tar_feat = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]
        
        ref_feats = ref_feats.reshape(-1, self.encoder.embed_dim)
        tar_feat = tar_feat.reshape(-1, self.encoder.embed_dim)

        # Normalization
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

        threshold = 0.8
        selected_mask = topk_similarities > threshold
        points_matched_inds_set = topk_indices[selected_mask]

        # 将patch索引转换成像素坐标点
        points_matched_inds_set_w = points_matched_inds_set % (self.encoder_feat_size)
        points_matched_inds_set_h = points_matched_inds_set // (self.encoder_feat_size)

        # 得到像素坐标
        idxs_mask_set_x = (points_matched_inds_set_w * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()
        idxs_mask_set_y = (points_matched_inds_set_h * self.encoder.patch_size + self.encoder.patch_size // 2).tolist()

        points_matched = []
        for x, y in zip(idxs_mask_set_x, idxs_mask_set_y):
            if int(x) < self.input_size[1] and int(y) < self.input_size[0]:
                points_matched.append([int(x), int(y)])

        points = np.array(points_matched)
        print(f'Matched points: {len(points)}')

        return points

    def clustering(self, points):
        num_centers = min(self.num_centers, len(points))
        flag = True
        while flag:
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            id, fre = torch.unique(cluster_assignment, return_counts=True)
            if id.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers

    def create_center_boxes(self, centers):
        # Using center to create bouding box
        boxes = []
        half_size = self.box_size // 2
        
        for center in centers:
            x, y = center[0], center[1]
            
            # 创建以 center 为中心的 box
            x_min = max(0, x - half_size)
            y_min = max(0, y - half_size)
            x_max = min(self.input_size[1] - 1, x + half_size)
            y_max = min(self.input_size[0] - 1, y + half_size)
            
            box = np.array([x_min, y_min, x_max, y_max])
            boxes.append(box)
        
        return boxes

    def visualize_boxes(self, target_image, centers, boxes, save_path=None):
        """可视化聚类中心和对应的 boxes"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # 转换图像格式
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img_display)
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime', 'brown', 'navy', 'teal', 'maroon', 'olive', 'coral', 'indigo', 'gold', 'crimson', 'violet']
        
        for i, (center, box) in enumerate(zip(centers, boxes)):
            color = colors[i % len(colors)]
            
            # 绘制 center
            ax.scatter(center[0], center[1], s=100, c=color, marker='*', 
                      edgecolors='white', linewidth=2, zorder=3)
            
            # 绘制 box
            rect = patches.Rectangle(
                (box[0], box[1]),  # (x, y) 左下角
                box[2] - box[0],   # width
                box[3] - box[1],   # height
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(center[0] + 5, center[1] - 5, f'{i+1}', color=color, 
                   fontsize=12, fontweight='bold')
        
        ax.set_title(f'Clustering Centers and Boxes (Total: {len(centers)})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Boxes visualization saved to {save_path}")
        
        plt.close()

    def visualize_merged_mask(self, target_image, merged_mask, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 转换图像格式
        if isinstance(target_image, torch.Tensor):
            img_display = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
        else:
            img_display = target_image
        
        # 转换 merged mask 格式
        if isinstance(merged_mask, torch.Tensor):
            mask_display = merged_mask.cpu().numpy()
        else:
            mask_display = merged_mask

        # 转换 true mask 格式
        if self.true_mask is not None:
            if isinstance(self.true_mask, torch.Tensor):
                true_mask_display = self.true_mask.squeeze().cpu().numpy()
            else:
                true_mask_display = self.true_mask

        # 转换partial mask (input)
        if hasattr(self, 'ref_masks_original') and self.ref_masks_original is not None:
            if isinstance(self.ref_masks_original, torch.Tensor):
                # ref_masks_original shape: [nshot, 1, H, W] 或 [1, nshot, H, W]
                partial_mask_display = self.ref_masks_original.squeeze().cpu().numpy()
                # 如果有多个 shot，取第一个或合并
                if len(partial_mask_display.shape) == 3:
                    partial_mask_display = partial_mask_display[0]  # 取第一个
            else:
                partial_mask_display = self.ref_masks_original

        
        else:
            partial_mask_display = None
        colormap_dict = {
            'red': 'Reds',
            'green': 'Greens', 
            'blue': 'Blues',
            'yellow': 'YlOrBr',
            'pink': 'RdPu'
        }
        mask_color = colormap_dict.get(self.target_color, 'Reds')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Original Image with Partial Mask (input)
        axes[0].imshow(img_display)
        if partial_mask_display is not None:
            axes[0].imshow(partial_mask_display, alpha=0.5, cmap='Blues')
            axes[0].set_title(f'Partial Mask (Input)\nArea: {partial_mask_display.sum():.0f} pixels', 
                              fontsize=14, fontweight='bold')
        else:
            axes[0].set_title('Partial Mask (Not Available)', 
                              fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 1. Original Image with True Mask
        axes[1].imshow(img_display)
        if true_mask_display is not None:
            axes[1].imshow(true_mask_display, alpha=0.5, cmap='Greens')
            axes[1].set_title(f'Ground Truth\nArea: {true_mask_display.sum():.0f} pixels', 
                              fontsize=14, fontweight='bold')
        else:
            axes[1].set_title('Ground Truth (Not Available)', 
                              fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 2. Origianl Mask with Predict Mask
        axes[2].imshow(img_display)
        axes[2].imshow(mask_display, alpha=0.5, cmap=mask_color)

        title_text = f'Predicted Mask\nArea: {mask_display.sum():.0f} pixels'
        axes[2].set_title(title_text, fontsize=14, fontweight='bold')
        axes[2].axis('off')

        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Merged mask visualization saved to {save_path}")
        
        plt.close()

    def mask_generation(self, tar_img_np, center_boxes):
        # 可视化 boxes
        if hasattr(self, 'tar_img') and self.tar_img is not None:
            save_path = f'IDRID/{self.target_color}/boxes_{self.target_color}/centers_boxes_{self.image_counter:04d}.png'
            centers = np.array([[(b[0]+b[2])//2, (b[1]+b[3])//2] for b in center_boxes])
            self.visualize_boxes(self.tar_img, centers, center_boxes, save_path=save_path)
            
        # 准备所有 boxes (center boxes)
        all_boxes = center_boxes.copy()

        if len(all_boxes) == 0:
            print("No box information provided")
        
        # 使用 SAM 进行分割，只使用 box prompt
        # 设置图像
        self.predictor.set_image(tar_img_np)

        # input
        input_boxes = torch.tensor(all_boxes,dtype=torch.float32, device=self.device)

        # Coordinate change
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            input_boxes, tar_img_np.shape[:2]
        )

        # Predict
        masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

        masks = masks.squeeze(1)

        tar_masks_np = masks.cpu().numpy() > 0
    
        # 构造与原代码兼容的 tar_masks_ori 格式（用于可视化）
        tar_masks_ori = []
        for i in range(masks.shape[0]):
            tar_masks_ori.append({
                'segmentation': tar_masks_np[i],
                'area': tar_masks_np[i].sum(),
                'predicted_iou': scores[i].item() if scores is not None else 0.0
            })
        
        # 可视化 SAM 输出
        sam_save_path = f'IDRID/{self.target_color}/sam_{self.target_color}/sam_outputs_{self.image_counter:04d}.png'
        self.visualize_sam_outputs(self.tar_img, tar_masks_ori, tar_masks_np[:, None, ...], save_path=sam_save_path)
        
        # 直接合并所有 masks
        merged_mask = tar_masks_np.sum(0) > 0
        merged_mask = merged_mask[None, ...]  # 添加 batch 维度

        merged_save_path = f'IDRID/{self.target_color}/merged_{self.target_color}/merged_mask_{self.image_counter:04d}.png'
        self.visualize_merged_mask(self.tar_img, merged_mask[0], save_path=merged_save_path)
            
        self.image_counter += 1
        
        return torch.tensor(merged_mask, device=self.device, dtype=torch.float)

    def visualize_sam_outputs(self, target_image, tar_masks_ori, tar_masks, save_path=None):
        """可视化SAM输出的masks"""
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
        
        total_masks = len(tar_masks_ori)
        
        if total_masks == 0:
            print("No masks to visualize")
            return
        
        # 选择最多16个mask展示
        num_to_show = min(16, total_masks)
        if total_masks > 16:
            selected_indices = random.sample(range(total_masks), 16)
        else:
            selected_indices = list(range(total_masks))
        
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()
        
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
        
        fig.suptitle(f'SAM Output Masks (Box-Only) ({num_to_show}/{total_masks} shown)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"SAM masks visualization saved to {save_path}")
        
        plt.close()

    def clear(self):
        self.tar_img = None
        self.tar_img_np = None
        self.ref_imgs = None
        self.ref_masks_pool = None
        self.nshot = None
        self.encoder_img_size = None
        self.encoder_feat_size = None


def build_matcher_oss(args):
    """构建使用 box-only prompt 的 Matcher"""
    
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
    sam.eval()
    predictor = SamPredictor(sam)
    
    score_filter_cfg = {
        "emd": args.emd_filter,
        "purity": args.purity_filter,
        "coverage": args.coverage_filter,
        "score_filter": True,
        "score": args.deep_score_filter,
        "score_norm": args.deep_score_norm_filter,
        "topk_scores_threshold": args.topk_scores_threshold
    }

    target_color = getattr(args, 'target_color', 'blue')
    
    box_size= 15

    
    # if target_color in ['blue', 'green', 'yellow']:
    #     num_centers = 10
    #     box_size = 20

    # else:
    #     num_centers = 20
    #     box_size = 5

    # print(f"Target color: {target_color}, num_centers: {num_centers}, box_size: {box_size}")

    return Matcher(
        encoder=dinov2,
        predictor = predictor,
        # num_centers=num_centers,
        box_size=box_size,
        use_box=args.use_box,
        alpha=args.alpha,
        beta=args.beta,
        exp=args.exp,
        score_filter_cfg=score_filter_cfg,
        num_merging_mask=args.num_merging_mask,
        device=args.device,
        target_color=args.target_color
    )