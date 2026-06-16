# -*- coding: utf-8 -*-


import sys
import time
import os

import numpy as np
import cv2
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import (
    QBrush, QPainter, QPen, QPixmap, QKeySequence, QColor, QImage
)
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QShortcut, QLabel, QMessageBox,
    QProgressBar
)
from skimage import transform, io
from PIL import Image

# 导入 SAM 库
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. 配置区
# ==========================================
SAM_MODEL_TYPE = "vit_b"
SAM_CKPT_PATH = "/host/d/Data/pretrained_SAM_weights/sam_vit_b.pth"#r'./sam_vit_b.pth'
print(f"Using SAM checkpoint: {SAM_CKPT_PATH}")

# 冻结随机种子
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ==========================================
# 2. 模型加载
# ==========================================
print("Loading SAM model...")
try:
    sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    # 注意：如果在这里报错，说明路径不对，请检查路径
    predictor = None


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


# ==========================================
# 3. 后台工作线程 (防止卡死)
# ==========================================
class EmbeddingWorker(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, image, predictor_instance):
        super().__init__()
        self.image = image
        self.predictor = predictor_instance

    def run(self):
        try:
            if self.predictor is None:
                raise ValueError("Predictor not initialized (Model load failed?)")
            # 耗时操作
            self.predictor.set_image(self.image)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


# ==========================================
# 4. 主窗口
# ==========================================
class Window(QWidget):
    def __init__(self):
        super().__init__()

        # --- 数据状态 ---
        self.image_path = None
        self.img_3c = None
        self.committed_mask = None
        self.current_mask = None

        # --- 交互状态 ---
        self.mode = "point"
        self.points_xy = []
        self.labels = []
        self.curr_box = None
        self.is_mouse_down = False
        self.start_pos = (None, None)
        self.rect_item = None

        # --- 线程 ---
        self.worker = None

        # --- 界面布局 ---
        self.setup_ui()
        self.setWindowTitle("SAM Lesion Segmentor (Fixed)")

        # --- Scene ---
        # 统一 1024x1024 画布
        self.scene = QGraphicsScene(0, 0, 1024, 1024)
        self.view.setScene(self.scene)
        self.bg_item = None

        # 检查模型是否加载成功
        if predictor is None:
            QMessageBox.critical(self, "Error", "Failed to load SAM model.\nCheck console for details.")

    def setup_ui(self):
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 跑马灯模式
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        self.info_label = QLabel("Load an image to start.")
        vbox.addWidget(self.info_label)

        hbox = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_save = QPushButton("Save Result")
        self.btn_point = QPushButton("Point Mode")
        self.btn_box = QPushButton("Box Mode")
        self.btn_finish = QPushButton("Commit Object (Space)")
        self.btn_clear = QPushButton("Clear Current (Esc)")

        hbox.addWidget(self.btn_load)
        hbox.addWidget(self.btn_save)
        hbox.addWidget(self.btn_point)
        hbox.addWidget(self.btn_box)
        hbox.addWidget(self.btn_finish)
        hbox.addWidget(self.btn_clear)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # 绑定
        self.btn_load.clicked.connect(self.load_image_start)
        self.btn_save.clicked.connect(self.save_mask)
        self.btn_point.clicked.connect(lambda: self.set_mode("point"))
        self.btn_box.clicked.connect(lambda: self.set_mode("box"))
        self.btn_finish.clicked.connect(self.commit_object)
        self.btn_clear.clicked.connect(self.clear_current_interaction)

        # 快捷键
        QShortcut(QKeySequence("Space"), self).activated.connect(self.commit_object)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.clear_current_interaction)

    def set_mode(self, mode):
        self.mode = mode
        self.info_label.setText(f"Mode: {mode.upper()}")
        self.view.setDragMode(QGraphicsView.NoDrag)

    # ================= 异步加载逻辑 =================

    def load_image_start(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", ".", "Images (*.jpg *.png *.tif *.bmp)")
        if not path: return

        try:
            # 1. 读取并缩放 (主线程做，很快)
            img = io.imread(path)
            if img.ndim == 2: img = np.repeat(img[:, :, None], 3, axis=2)
            if img.shape[2] == 4: img = img[:, :, :3]

            # 强制缩放至 1024x1024
            self.img_3c = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(
                np.uint8)
            self.image_path = path

            # 2. UI 锁定
            self.info_label.setText("Calculating SAM Embedding... Please wait.")
            self.progress_bar.setVisible(True)
            self.btn_load.setEnabled(False)
            self.view.setEnabled(False)

            # 3. 启动线程计算 Embedding
            self.worker = EmbeddingWorker(self.img_3c, predictor)
            self.worker.finished_signal.connect(self.on_embedding_finished)
            self.worker.error_signal.connect(self.on_embedding_error)
            self.worker.start()

        except Exception as e:
            self.info_label.setText(f"Error reading image: {e}")

    def on_embedding_finished(self):
        """线程成功回调"""
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.view.setEnabled(True)

        self.info_label.setText("Ready. Left Click=FG, Right Click=BG.")

        # 初始化数据
        self.committed_mask = np.zeros((1024, 1024), dtype=np.uint8)
        self.current_mask = np.zeros((1024, 1024), dtype=np.uint8)

        self.clear_current_interaction()
        self._refresh_display()

        # ！！！关键：重新绑定事件！！！
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def on_embedding_error(self, error_msg):
        """线程失败回调"""
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.view.setEnabled(True)
        QMessageBox.critical(self, "SAM Error", f"Failed to calculate embedding:\n{error_msg}")

    # ================= 交互逻辑 (修复版) =================

    def _get_xy(self, ev):
        """
        【关键修复】
        ev 是 QGraphicsSceneMouseEvent，它带有 scenePos()。
        直接读取 scenePos 才是正确的 1024x1024 坐标。
        不要再用 view.mapToScene 了！
        """
        pos = ev.scenePos()
        x, y = pos.x(), pos.y()
        return max(0, min(1023, x)), max(0, min(1023, y))

    def mouse_press(self, ev):
        try:
            x, y = self._get_xy(ev)

            if self.mode == "point":
                # 左键=1(前景), 右键=0(背景)
                label = 0 if (ev.button() == Qt.RightButton) else 1
                self.points_xy.append([x, y])
                self.labels.append(label)

                # 画点反馈
                color = QColor("green") if label == 1 else QColor("red")
                self.scene.addEllipse(x - 4, y - 4, 8, 8, QPen(color), QBrush(color))

                # 触发推理
                self.run_inference()

            elif self.mode == "box":
                self.is_mouse_down = True
                self.start_pos = (x, y)
        except Exception as e:
            print(f"Click Error: {e}")

    def mouse_move(self, ev):
        if self.mode == "box" and self.is_mouse_down:
            x, y = self._get_xy(ev)
            if self.rect_item: self.scene.removeItem(self.rect_item)

            sx, sy = self.start_pos
            self.curr_box = [min(sx, x), min(sy, y), max(sx, x), max(sy, y)]
            self.rect_item = self.scene.addRect(*self.curr_box[0:2],
                                                self.curr_box[2] - self.curr_box[0],
                                                self.curr_box[3] - self.curr_box[1],
                                                QPen(QColor("blue")))

    def mouse_release(self, ev):
        if self.mode == "box" and self.is_mouse_down:
            self.is_mouse_down = False
            self.run_inference()

    # ================= 核心推理 =================

    def run_inference(self):
        if self.img_3c is None: return

        points = np.array(self.points_xy) if self.points_xy else None
        labels = np.array(self.labels) if self.labels else None
        box = np.array(self.curr_box) if self.curr_box else None

        if points is None and box is None: return

        try:
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=True
                )
        except Exception as e:
            print(f"Inference Error: {e}")
            return

        # 策略：如果只有点，选面积最小的；如果有框，选分数最高的
        if box is None:
            areas = np.sum(masks, axis=(1, 2))
            best_idx = np.argmin(areas)
            info = "Area (Smallest)"
        else:
            best_idx = np.argmax(scores)
            info = "Score (Highest)"

        self.current_mask = masks[best_idx].astype(np.uint8)
        self.info_label.setText(f"Inference: {info}, IoU: {scores[best_idx]:.3f}")
        self._refresh_display()

    # ================= 状态管理 =================

    def commit_object(self):
        """确认当前分割"""
        if self.current_mask is None or np.max(self.current_mask) == 0:
            return

        # 叠加到永久层
        self.committed_mask = np.maximum(self.committed_mask, self.current_mask * 255)
        self.info_label.setText("Object Committed. Ready for next.")
        self.clear_current_interaction()
        self._refresh_display()

    def clear_current_interaction(self):
        """清除当前未确认的交互"""
        self.points_xy = []
        self.labels = []
        self.curr_box = None
        self.current_mask = np.zeros((1024, 1024), dtype=np.uint8)
        self.scene.clear()
        self.rect_item = None
        self._refresh_display()

    def _refresh_display(self):
        if self.img_3c is None: return

        # 1. 基础底图
        display_img = self.img_3c.copy()

        # 2. 叠加已确认 (Committed) - 蓝色
        if np.max(self.committed_mask) > 0:
            color_mask = np.zeros_like(display_img)
            color_mask[self.committed_mask > 0] = [0, 0, 255]
            display_img = cv2.addWeighted(display_img, 1.0, color_mask, 0.4, 0)

        # 3. 叠加当前 (Current) - 红色
        if self.current_mask is not None and np.max(self.current_mask) > 0:
            color_mask = np.zeros_like(display_img)
            color_mask[self.current_mask > 0] = [255, 0, 0]
            display_img = cv2.addWeighted(display_img, 1.0, color_mask, 0.5, 0)

        # 4. 显示
        pixmap = np2pixmap(display_img)
        self.scene.clear()
        self.bg_item = self.scene.addPixmap(pixmap)

        # 5. 重绘点 (因为 clear 擦掉了)
        for pt, label in zip(self.points_xy, self.labels):
            c = QColor("green") if label == 1 else QColor("red")
            self.scene.addEllipse(pt[0] - 4, pt[1] - 4, 8, 8, QPen(c), QBrush(c))

        if self.curr_box:
            bx = self.curr_box
            self.rect_item = self.scene.addRect(bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1], QPen(QColor("blue")))

    def save_mask(self):
        if not self.image_path: return

        # --- 修复开始：使用 os.path 智能处理路径 ---
        root, ext = os.path.splitext(self.image_path)
        out = f"{root}_sam_result.png"  # 强制保存为 png 格式，防止 jpg 压缩损失
        # ---------------------------------------

        try:
            # 合并：已确认的蓝色区域 + 当前红色的区域
            final = np.maximum(self.committed_mask, self.current_mask * 255)

            # 确保是二值图（黑白），如果你想要彩色的可以不转
            # final = (final > 0).astype(np.uint8) * 255

            io.imsave(out, final)
            QMessageBox.information(self, "Saved", f"Mask saved successfully to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save mask:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())