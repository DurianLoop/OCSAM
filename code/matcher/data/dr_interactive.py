"""
DR Dataset with Interactive Component Selection
数据路径: datapath/DR_Training_Set/Fundus Images/ 和 Combined Masks/
工作流程: Load mask → Extract target_color → CCA → Interactive selection
"""

import os
from os import path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import random
from typing import List, Tuple
from .component_selector import get_component_mask, ComponentSelector as ComponentSelector


class DatasetDR(Dataset):
    """DR (Diabetic Retinopathy) Dataset with Interactive Component Selection"""
    
    def __init__(
        self, 
        datapath, 
        transform, 
        use_original_imgsize=False, 
        target_color='EX',
        interactive_mode=False
    ):
        """
        Args:
            datapath: 数据集根目录
            transform: 图像变换
            use_original_imgsize: 是否使用原始图像尺寸
            target_color: 目标颜色类别 (EX, HE, MA, SE, whole)
            interactive_mode: 是否启用交互式选择模式
        """
        self.datapath = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.target_color = target_color
        self.interactive_mode = interactive_mode
        
        # 设置图像和标注路径
        self.img_path = path.join(datapath, 'DR_Training_Set', 'Fundus Images')
        self.mask_path = path.join(datapath, 'DR_Training_Set', 'Combined Masks')
        
        # 检查路径是否存在
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"Image path not found: {self.img_path}")
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask path not found: {self.mask_path}")
        
        # 获取所有图像文件
        self.img_metadata = self.build_img_metadata()
        
        print(f"\n{'='*70}")
        print(f"DatasetDR Initialized:")
        print(f"  • Target color: {target_color}")
        print(f"  • Mode: {'🎮 Interactive' if interactive_mode else '🎲 Random'}")
        print(f"  • Total images: {len(self.img_metadata)}")
        print(f"  • Image path: {self.img_path}")
        print(f"  • Mask path: {self.mask_path}")
        print('='*70)
        
    def build_img_metadata(self):
        """构建图像元数据列表"""
        img_metadata = []
        
        # 遍历图像文件
        img_files = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]
        
        for img_name in sorted(img_files):
            # 对应的 mask 文件
            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}.png"
            mask_path_full = path.join(self.mask_path, mask_name)
            
            # 检查 mask 是否存在
            if os.path.exists(mask_path_full):
                img_metadata.append({
                    'image_name': img_name,
                    'mask_name': mask_name,
                    'image_path': path.join(self.img_path, img_name),
                    'mask_path': mask_path_full
                })
        
        if len(img_metadata) == 0:
            print(f"Warning: No valid image-mask pairs found!")
            print(f"  Checked image path: {self.img_path}")
            print(f"  Checked mask path: {self.mask_path}")
        
        return img_metadata
    
    def _load_mask(self, mask_file, target_color='EX'):
        """
        Load and process mask file with target color extraction
        
        Args:
            mask_file: Path to mask file
            target_color: Color to extract ('EX', 'HE', 'MA', 'SE', 'whole')
            
        Returns:
            np.ndarray: Binary mask array [H, W]
        """
        try:
            mask = Image.open(mask_file)
            mask_array = np.array(mask)
            
            # Convert to binary mask based on target_color
            if len(mask_array.shape) == 3:
                # EX (Exudates) - Yellow/bright regions
                if target_color == 'EX':
                    color_mask = (mask_array[:,:,0] > 120) & \
                                 (mask_array[:,:,1] > 120) & \
                                 (mask_array[:,:,2] < 50)
                    binary_mask = color_mask.astype(np.uint8)
                
                # HE (Hemorrhages) - Red regions
                elif target_color == 'HE':
                    color_mask = (mask_array[:,:,0] > 150) & \
                                 (mask_array[:,:,1] < 50) & \
                                 (mask_array[:,:,2] < 50)
                    binary_mask = color_mask.astype(np.uint8)
                
                # MA (Microaneurysms) - Green regions
                elif target_color == 'MA':
                    color_mask = (mask_array[:,:,1] > 150) & \
                                 (mask_array[:,:,0] < 50) & \
                                 (mask_array[:,:,2] < 50)
                    binary_mask = color_mask.astype(np.uint8)
                
                # SE (Soft Exudates) - Blue regions
                elif target_color == 'SE':
                    color_mask = (mask_array[:,:,2] > 100) & \
                                 (mask_array[:,:,0] < 100) & \
                                 (mask_array[:,:,1] < 100)
                    binary_mask = color_mask.astype(np.uint8)
                
                # Whole - any foreground
                elif target_color == 'whole':
                    blue_mask = (mask_array[:,:,2] > 100) & \
                                (mask_array[:,:,0] < 100) & \
                                (mask_array[:,:,1] < 100)
                    
                    green_mask = (mask_array[:,:,1] > 150) & \
                                 (mask_array[:,:,0] < 50) & \
                                 (mask_array[:,:,2] < 50)
                    
                    red_mask = (mask_array[:,:,0] > 150) & \
                               (mask_array[:,:,1] < 50) & \
                               (mask_array[:,:,2] < 50)
                    
                    yellow_mask = (mask_array[:,:,0] > 120) & \
                                  (mask_array[:,:,1] > 120) & \
                                  (mask_array[:,:,2] < 50)
                    
                    binary_mask = (blue_mask | green_mask | red_mask | yellow_mask).astype(np.uint8)
                
                else:
                    # Default: any non-black pixel
                    binary_mask = (np.any(mask_array > 10, axis=2)).astype(np.uint8)
            else:
                binary_mask = (mask_array > 127).astype(np.uint8)
            
            return binary_mask
            
        except Exception as e:
            print(f"Error loading mask {mask_file}: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def _get_components(self, mask: np.ndarray, min_area: int = 50) -> Tuple[np.ndarray, int, dict]:
        """提取 mask 中的 connected components
        
        Args:
            mask: Binary mask (H, W)
            min_area: 最小 component 面积阈值
            
        Returns:
            components: Component labeled mask (H, W)
            num_components: 有效 component 数量
            stats: Component 统计信息
        """
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # 过滤小面积的 components
        valid_components = np.zeros_like(labels)
        valid_id = 1
        component_info = {}
        
        for i in range(1, num_labels):  # 跳过背景 (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_components[labels == i] = valid_id
                component_info[valid_id] = {
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': stats[i, :4]  # x, y, w, h
                }
                valid_id += 1
        
        num_valid = valid_id - 1
        
        return valid_components, num_valid, component_info
    
    def _select_components_interactive(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        components: np.ndarray, 
        num_components: int,
        image_name: str
    ) -> List[int]:
        """交互式选择 components"""
        
        # 创建交互式选择器
        selector = ComponentSelector(
            image=image,
            mask=mask,
            components=components,
            num_components=num_components,
            max_selections=2
        )
        
        print(f"\n{'='*70}")
        print(f"🎮 Interactive Selection Mode")
        print(f"Image: {image_name}")
        print(f"Target Color: {self.target_color}")
        print(f"Total components: {num_components}")
        print('='*70)
        
        # 显示交互界面
        selected = selector.select()
        
        return selected
    
    def _select_components_random(self, num_components: int) -> List[int]:
        """随机选择 1-2 个 components"""
        num_select = random.randint(1, min(2, num_components))
        selected = random.sample(range(1, num_components + 1), num_select)
        return selected
    
    def __len__(self):
        return len(self.img_metadata)
    
    def __getitem__(self, idx):
        """获取数据样本
        
        工作流程:
        1. Load combined mask (多种颜色)
        2. Extract target_color → specific_mask (只包含该颜色)
        3. CCA on specific_mask → 获取该颜色的 components
        4. Select components → 只在 specific_mask 的 components 中选择
        """
        metadata = self.img_metadata[idx]
        
        # 读取图像
        image = Image.open(metadata['image_path']).convert('RGB')
        
        # Load mask and extract target_color
        mask_np = self._load_mask(metadata['mask_path'], self.target_color)
        
        # 如果 mask 为空，返回空 mask
        if mask_np.sum() == 0:
            print(f"\n Warning: Empty mask for {metadata['image_name']} with color {self.target_color}")
            support_mask = mask_np
            selected_components = []
            num_components = 0
        else:
            # 对 specific_mask 进行 CCA
            components, num_components, component_info = self._get_components(mask_np)
            
            # 如果没有有效的 components，返回原始 mask
            if num_components == 0:
                print(f"\n Warning: No valid components found in {metadata['image_name']}")
                support_mask = mask_np
                selected_components = []
            else:
                # Step 3: 根据模式选择 components
                image_np = np.array(image)
                
                if self.interactive_mode:
                    # 交互式选择
                    selected_components = self._select_components_interactive(
                        image=image_np,
                        mask=mask_np,
                        components=components,
                        num_components=num_components,
                        image_name=metadata['image_name']
                    )
                else:
                    # 随机选择
                    selected_components = self._select_components_random(num_components)
                
                # Step 4: 生成 partial mask
                if selected_components:
                    support_mask = get_component_mask(mask_np, selected_components, components)
                else:
                    support_mask = mask_np
        
        # 打印信息
        print(f"\n{'='*70}")
        print(f"Image: {metadata['image_name']}")
        print(f"  • Target color: {self.target_color}")
        print(f"  • Total components: {num_components}")
        print(f"  • Selected components: {selected_components}")
        if mask_np.sum() > 0:
            coverage = support_mask.sum() / mask_np.sum() * 100
            print(f"  • Partial mask coverage: {coverage:.1f}%")
        print('='*70)
        
        # 应用 transform
        if self.transform:
            image = self.transform(image)
            mask = self.transform(Image.fromarray(mask_np * 255))
            support_mask = self.transform(Image.fromarray(support_mask * 255))
        
        # 转换为 torch tensor
        mask = (mask > 0).float()
        support_mask = (support_mask > 0).float()

        if mask.dim() == 3:
            mask = mask[0]
        if support_mask.dim() == 3:
            support_mask = support_mask[0]        
        
        # 构建返回的 batch
        batch = {
            'query_img': image,
            'query_mask': mask,
            'full_mask': mask,  # 完整的 ground truth
            'support_imgs': image.unsqueeze(0),  # 1-shot: 自身作为 support
            'support_masks': support_mask.unsqueeze(0),
            'class_id': self.target_color,
            'query_mask_file': metadata['mask_name'],
            'support_mask_files': [metadata['mask_name']],
            # 额外信息
            'image_name': metadata['image_name'],
            'num_components': num_components,
            'selected_components': selected_components,
        }
        
        return batch