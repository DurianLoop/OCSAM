import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image
import numpy as np
from glob import glob
import random
from scipy.ndimage import label
from skimage.measure import regionprops


class DatasetMonuSeg(Dataset):
    """
    MonuSeg Dataset for Partial-to-Full Segmentation
    
    Task: Given an image with partial foreground mask (selected by CCA),
          predict the complete foreground mask
    
    Simplified Design:
        - No fold/split complexity
        - Precompute all partial masks during initialization
        - Direct indexing in __getitem__ (no random sampling)
        - Compatible with existing evaluation framework
    """
    
    def __init__(self, datapath, transform, use_original_imgsize):
        """
        Initialize MonuSeg dataset with simplified structure
        
        Args:
            datapath: Root directory containing MonuSeg dataset
            transform: Image transformations
            use_original_imgsize: Whether to use original image size
        """
        # Basic configuration
        self.benchmark = 'MonuSeg'
        self.nclass = 1  # Single class (nuclei)
        
        # Image processing
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.img_size = 518  # Default size
        
        # Data paths
        self.img_path = os.path.join(datapath, 'kmms', 'images')
        self.mask_path = os.path.join(datapath, 'kmms', 'masks')
        
        # Build dataset with precomputed partial masks
        self.data_list = []
        self._build_dataset()
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n{'='*60}")
        print(f"MonuSeg Dataset - Partial-to-Full Segmentation (Simplified)")
        print(f"  Benchmark: {self.benchmark}")
        print(f"  Total samples: {len(self.data_list)}")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Image path: {self.img_path}")
        print(f"  Mask path: {self.mask_path}")
        print(f"\n  Task Description:")
        print(f"    INPUT:  Image + Partial mask (1-2 components from CCA)")
        print(f"    OUTPUT: Predict full mask (all foreground)")
        print(f"    Method: Connected Component Analysis (top 20%)")
        print(f"{'='*60}\n")
    
    def _build_dataset(self):
        """
        Build dataset with precomputed partial masks
        
        This method:
        1. Finds all image-mask pairs
        2. Loads full masks
        3. Performs CCA to generate partial masks
        4. Stores everything in memory for fast access
        """
        # Find all valid image-mask pairs
        metadata = self._build_metadata()
        
        if len(metadata) == 0:
            print("Warning: No valid data found!")
            return
        
        print(f"Processing {len(metadata)} images and generating partial masks...")
        
        for i, data in enumerate(metadata):
            try:
                # Load full mask (ground truth)
                full_mask = self._load_mask(data['mask_file'])
                
                # Generate partial mask using CCA
                partial_mask = self._connected_component_analysis(full_mask)
                
                # Store all information
                self.data_list.append({
                    'img_file': data['img_file'],
                    'mask_file': data['mask_file'],
                    'img_name': data['img_name'],
                    'full_mask': full_mask,        # Precomputed
                    'partial_mask': partial_mask,  # Precomputed
                    'class_id': 0
                })
                    
            except Exception as e:
                print(f"Error processing {data['img_name']}: {e}")
                continue
        
        print(f"Dataset built successfully: {len(self.data_list)} samples ready")
    
    def _build_metadata(self):
        """
        Build metadata for the dataset
        
        Returns:
            list: List of valid image-mask pairs
        """
        if not os.path.exists(self.img_path):
            print(f"Error: Image path {self.img_path} does not exist")
            return []
        
        if not os.path.exists(self.mask_path):
            print(f"Error: Mask path {self.mask_path} does not exist")
            return []
        
        # Find all image files
        img_extensions = ['*.png', '*.tif']
        
        img_files = []
        for ext in img_extensions:
            found = glob(os.path.join(self.img_path, ext))
            img_files.extend(found)
        
        if len(img_files) == 0:
            print(f"Error: No image files found in {self.img_path}")
            return []
        
        print(f"Found {len(img_files)} image files")
        
        # Build valid image-mask pairs
        metadata = []
        
        for img_file in sorted(img_files):
            img_name = os.path.basename(img_file)
            base_name = os.path.splitext(img_name)[0]
            
            # Find corresponding mask
            mask_file = self._find_mask_file(base_name)
            
            if mask_file and self._is_valid_mask(mask_file):
                metadata.append({
                    'img_file': img_file,
                    'mask_file': mask_file,
                    'img_name': img_name,
                    'base_name': base_name,
                })
        
        print(f"Valid image-mask pairs: {len(metadata)}")
        
        return metadata
    
    def _find_mask_file(self, base_name):
        """Find mask file for given base name"""
        mask_extensions = ['.png']
        
        for ext in mask_extensions:
            mask_file = os.path.join(self.mask_path, base_name + ext)
            if os.path.exists(mask_file):
                return mask_file
        return None
    
    def _is_valid_mask(self, mask_file):
        """Check if mask contains foreground pixels"""
        try:
            mask = Image.open(mask_file)
            mask_array = np.array(mask)
            
            # Convert to grayscale if needed
            if len(mask_array.shape) == 3:
                mask_array = np.mean(mask_array, axis=2)
            
            # Check for foreground pixels
            return np.any(mask_array > 5)
        except Exception as e:
            print(f"Error reading mask {mask_file}: {e}")
            return False
    
    def _load_mask(self, mask_file):
        """
        Load and process binary mask file
        
        Args:
            mask_file: Path to mask file
            
        Returns:
            torch.Tensor: Binary mask tensor [H, W]
        """
        try:
            mask = Image.open(mask_file)
            mask_array = np.array(mask)
            
            # Convert to grayscale if RGB/RGBA
            if len(mask_array.shape) == 3:
                if mask_array.shape[2] == 4:
                    mask_array = mask_array[:, :, :3]
                mask_array = np.mean(mask_array, axis=2)
            
            # Convert to binary mask (threshold at 127)
            binary_mask = (mask_array > 127).astype(np.float32)
            
            # Handle empty masks
            if binary_mask.sum() == 0:
                print(f"Warning: Empty mask in {mask_file}, creating small central region")
                h, w = binary_mask.shape
                center_h, center_w = h // 2, w // 2
                size = max(5, min(h, w) // 40)
                binary_mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            
            return torch.tensor(binary_mask, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading mask {mask_file}: {e}")
            # Return a small central mask as fallback
            h, w = 512, 512
            binary_mask = np.zeros((h, w), dtype=np.float32)
            center_h, center_w = h // 2, w // 2
            size = 10
            binary_mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            return torch.tensor(binary_mask, dtype=torch.float32)
    
    def _connected_component_analysis(self, mask):
        """
        Perform connected component analysis and select 1-2 components as PARTIAL mask
        
        Args:
            mask: Full binary mask tensor [H, W] (complete foreground)
            
        Returns:
            torch.Tensor: Partial mask [H, W] (selected components as prompt)
        """
        try:
            # Convert to numpy for processing
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # Perform connected component analysis
            labeled_mask, num_components = label(mask_np)
            
            if num_components == 0:
                print("Warning: No connected components found")
                return mask
            
            # Get region properties
            regions = regionprops(labeled_mask)
            
            if len(regions) == 0:
                print("Warning: No regions found")
                return mask
            
            # Sort components by area (descending)
            component_areas = [(region.area, region.label) for region in regions]
            component_areas.sort(key=lambda x: x[0], reverse=True)
            
            # Select top 20% components by area
            top_20_percent_count = max(1, int(len(component_areas) * 0.2))
            top_components = component_areas[:top_20_percent_count]
            
            # Randomly select 1-2 components from top 20%
            num_to_select = min(random.randint(1, 2), len(top_components))
            selected_components = random.sample(top_components, num_to_select)
            
            # Create combined mask
            combined_mask = np.zeros_like(mask_np, dtype=np.float32)
            
            for area, label_id in selected_components:
                component_mask = (labeled_mask == label_id).astype(np.float32)
                combined_mask = np.maximum(combined_mask, component_mask)
            
            # Convert back to tensor
            selected_mask_tensor = torch.tensor(combined_mask, dtype=torch.float32)
            
            return selected_mask_tensor
            
        except Exception as e:
            print(f"Error in connected component analysis: {e}")
            # Return a small central region as fallback
            h, w = mask.shape
            fallback_mask = torch.zeros_like(mask)
            center_h, center_w = h // 2, w // 2
            size = max(5, min(h, w) // 40)
            fallback_mask[center_h-size:center_h+size, center_w-size:center_w+size] = 1
            return fallback_mask
    
    def __len__(self):
        """Dataset size"""
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Get item by direct indexing (no random sampling)
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict: Batch dictionary compatible with evaluation framework
        """
        try:
            # Get data by index
            data = self.data_list[idx]
            
            # Load image
            img = Image.open(data['img_file']).convert('RGB')
            org_img_size = img.size  # (W, H)
            
            # Apply transforms
            if self.transform is not None:
                img_transformed = self.transform(img)
            else:
                img_resized = img.resize((self.img_size, self.img_size), 
                                        Image.Resampling.LANCZOS)
                img_transformed = transforms.ToTensor()(img_resized)
            
            # Get precomputed masks
            full_mask = data['full_mask']
            partial_mask = data['partial_mask']
            
            # Resize masks
            full_mask_resized = self._resize_mask(full_mask, (self.img_size, self.img_size))
            partial_mask_resized = self._resize_mask(partial_mask, (self.img_size, self.img_size))
            
            # Create batch dictionary (compatible with FSSDataset framework)
            batch = {
                # === Query: Target to predict ===
                'query_img': img_transformed,              # [3, H, W] - Image to segment
                'query_mask': full_mask_resized,           # [H, W] - For evaluation
                'full_mask': full_mask_resized,            # [H, W] - Ground truth for Matcher
                'query_name': data['img_name'],            # str
                'query_mask_file': data['mask_file'],      # str
                'query_class': torch.tensor(0),            # scalar - class ID
                'org_query_imsize': org_img_size,          # (W, H)
                
                # === Support: Reference for guidance ===
                'support_imgs': img_transformed.unsqueeze(0),       # [1, 3, H, W] - Same image
                'support_masks': partial_mask_resized.unsqueeze(0), # [1, H, W] - Partial mask
                'support_names': [data['img_name']],        # list of str
                'support_mask_files': [data['mask_file']],  # list of str
                'support_classes': torch.tensor([0]),       # [1] - class IDs
                
                # === Compatibility fields ===
                'class_id': torch.tensor(0),
                'class_ids': [0],
            }
            
            return batch
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            # Return a dummy batch to prevent interruption
            dummy_img = torch.zeros(3, self.img_size, self.img_size)
            dummy_mask = torch.zeros(self.img_size, self.img_size)
            
            batch = {
                'query_img': dummy_img,
                'query_mask': dummy_mask,
                'full_mask': dummy_mask, # as true mask for Matcher
                'query_name': 'dummy',
                'query_mask_file': 'dummy',
                'query_class': torch.tensor(0),
                'org_query_imsize': (self.img_size, self.img_size),
                'support_imgs': dummy_img.unsqueeze(0),
                'support_masks': dummy_mask.unsqueeze(0),
                'support_names': ['dummy'],
                'support_mask_files': ['dummy'],
                'support_classes': torch.tensor([0]),
                'class_id': torch.tensor(0),
                'class_ids': [0]
            }
            return batch
    
    def _resize_mask(self, mask, size):
        """
        Resize mask to target size
        
        Args:
            mask: Mask tensor [H, W]
            size: Target size (H, W)
            
        Returns:
            Resized mask tensor [H, W]
        """
        try:
            # Add batch and channel dimensions
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Resize using nearest neighbor to preserve binary nature
            mask_resized = F.interpolate(
                mask.float(),
                size=size,
                mode='nearest'
            )
            
            # Remove extra dimensions and binarize
            mask_resized = mask_resized.squeeze()  # [H, W]
            mask_resized = (mask_resized > 0.5).float()
            
            return mask_resized
            
        except Exception as e:
            print(f"Error resizing mask: {e}")
            return torch.zeros(size, dtype=torch.float32)