import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import cv2
from typing import List


class ComponentSelector:
    """交互式 component 选择器 - 改进版"""
    
    def __init__(self, image: np.ndarray, mask: np.ndarray, components: np.ndarray, 
                 num_components: int, max_selections: int = 2):
        """
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            components: Component labeled mask (H, W), 每个像素值代表其 component ID
            num_components: 总 component 数量
            max_selections: 最多选择的 component 数量
        """
        self.image = image
        self.mask = mask
        self.components = components
        self.num_components = num_components
        self.max_selections = max_selections
        
        # 存储选中的 component IDs
        self.selected_components = []
        
        # 为每个 component 生成颜色
        self.colors = self._generate_colors(num_components)
        
        # 预计算每个 component 的轮廓和中心
        self.contours = {}
        self.centers = {}
        self._precompute_contours()
        
        # 创建可视化
        self.fig = None
        self.ax = None
        
    def _generate_colors(self, n: int) -> np.ndarray:
        """生成 n 个不同的颜色"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(rgb)
        return np.array(colors)
    
    def _precompute_contours(self):
        """预计算每个 component 的轮廓和中心"""
        for comp_id in range(1, self.num_components + 1):
            mask_i = (self.components == comp_id).astype(np.uint8)
            
            if mask_i.sum() > 0:
                # 找轮廓
                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.contours[comp_id] = contours
                
                # 计算中心
                y_coords, x_coords = np.where(mask_i)
                self.centers[comp_id] = (x_coords.mean(), y_coords.mean())
    
    def _update_display(self):
        """更新显示"""
        self.ax.clear()
        
        # 显示原图
        self.ax.imshow(self.image)
        
        # 创建叠加层
        overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.float32)
        
        # 绘制每个 component
        for comp_id in range(1, self.num_components + 1):
            if comp_id not in self.contours:
                continue
                
            color = self.colors[comp_id - 1]
            is_selected = comp_id in self.selected_components
            
            # 选中的 component: 填充 + 粗轮廓
            if is_selected:
                # 填充选中区域
                mask_i = (self.components == comp_id)
                overlay[mask_i, :3] = color
                overlay[mask_i, 3] = 0.5  # 半透明
                
                # 画粗轮廓
                for contour in self.contours[comp_id]:
                    contour = contour.squeeze()
                    if len(contour.shape) == 1:
                        contour = contour.reshape(1, 2)
                    if len(contour) > 1:
                        self.ax.plot(contour[:, 0], contour[:, 1], 
                                   color=color, linewidth=3, linestyle='-')
                        # 闭合轮廓
                        self.ax.plot([contour[-1, 0], contour[0, 0]], 
                                   [contour[-1, 1], contour[0, 1]], 
                                   color=color, linewidth=3, linestyle='-')
            else:
                # 未选中: 只画细轮廓
                for contour in self.contours[comp_id]:
                    contour = contour.squeeze()
                    if len(contour.shape) == 1:
                        contour = contour.reshape(1, 2)
                    if len(contour) > 1:
                        self.ax.plot(contour[:, 0], contour[:, 1], 
                                   color=color, linewidth=1.5, linestyle='-', alpha=0.7)
                        self.ax.plot([contour[-1, 0], contour[0, 0]], 
                                   [contour[-1, 1], contour[0, 1]], 
                                   color=color, linewidth=1.5, linestyle='-', alpha=0.7)
            
            # 添加编号标签
            if comp_id in self.centers:
                cx, cy = self.centers[comp_id]
                
                if is_selected:
                    # 选中: 白色文字，彩色背景
                    self.ax.text(cx, cy, str(comp_id),
                               color='white', fontsize=10, fontweight='bold',
                               ha='center', va='center',
                               bbox=dict(boxstyle='circle,pad=0.3', 
                                       facecolor=color, edgecolor='white',
                                       linewidth=2, alpha=0.9))
                else:
                    # 未选中: 小号灰色文字
                    self.ax.text(cx, cy, str(comp_id),
                               color='white', fontsize=8,
                               ha='center', va='center',
                               bbox=dict(boxstyle='circle,pad=0.2', 
                                       facecolor='gray', alpha=0.6))
        
        # 显示叠加层
        self.ax.imshow(overlay)
        
        # 标题
        if self.selected_components:
            title = f"✓ Selected: {self.selected_components} ({len(self.selected_components)}/{self.max_selections})"
        else:
            title = f"Click to select components (0/{self.max_selections})"
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def _on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= y < self.components.shape[0] and 0 <= x < self.components.shape[1]:
            comp_id = int(self.components[y, x])
            
            if comp_id > 0:
                if comp_id in self.selected_components:
                    self.selected_components.remove(comp_id)
                    print(f"❌ Deselected Component {comp_id}")
                else:
                    if len(self.selected_components) < self.max_selections:
                        self.selected_components.append(comp_id)
                        print(f"✓ Selected Component {comp_id}")
                    else:
                        print(f"⚠️  Maximum {self.max_selections} components allowed!")
                        return
                
                self._update_display()
    
    def _on_confirm(self, event):
        """确认选择"""
        if len(self.selected_components) == 0:
            print("⚠️  Please select at least one component!")
            return
        print(f"\n✓ Confirmed: {self.selected_components}")
        plt.close(self.fig)
    
    def _on_reset(self, event):
        """重置选择"""
        self.selected_components = []
        print("🔄 Reset")
        self._update_display()
    
    def select(self) -> List[int]:
        """显示交互界面并返回选中的 component IDs"""
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = plt.subplot(111)
        
        self._update_display()
        
        # 按钮
        ax_confirm = plt.axes([0.35, 0.02, 0.12, 0.04])
        btn_confirm = Button(ax_confirm, '✓ Confirm', color='lightgreen', hovercolor='green')
        btn_confirm.on_clicked(self._on_confirm)
        
        ax_reset = plt.axes([0.53, 0.02, 0.12, 0.04])
        btn_reset = Button(ax_reset, '↺ Reset', color='lightcoral', hovercolor='red')
        btn_reset.on_clicked(self._on_reset)
        
        # 点击事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # 说明
        instructions = f"Click on components to select (max {self.max_selections}) | Total: {self.num_components} components"
        self.fig.text(0.5, 0.97, instructions, fontsize=11, ha='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.show()
        
        return self.selected_components


def get_component_mask(mask: np.ndarray, selected_components: List[int], 
                       components: np.ndarray) -> np.ndarray:
    """根据选中的 component IDs 生成 partial mask"""
    partial_mask = np.zeros_like(mask)
    for comp_id in selected_components:
        partial_mask[components == comp_id] = 1
    return partial_mask


# ============ 测试用 ============
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    # 模拟图像
    image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    # 模拟 components (5个随机区域)
    components = np.zeros((512, 512), dtype=np.int32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    
    for i in range(1, 6):
        cx, cy = np.random.randint(100, 400, 2)
        rr, cc = np.ogrid[:512, :512]
        circle = ((rr - cy)**2 + (cc - cx)**2) < (30 + i*10)**2
        components[circle] = i
        mask[circle] = 1
    
    # 测试选择器
    selector = ComponentSelector(
        image=image,
        mask=mask,
        components=components,
        num_components=5,
        max_selections=2
    )
    
    selected = selector.select()
    print(f"Final selection: {selected}")