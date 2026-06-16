"""Tiny OpenCV compatibility layer for this Windows reproduction workspace.

The conda OpenCV DLL build is not reliable on this machine. The project only
uses a small subset of cv2 APIs in the demo/evaluation path, so we provide
pure-Python/numpy replacements here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

INTER_AREA = 3
RETR_CCOMP = 2
RETR_EXTERNAL = 0
CHAIN_APPROX_NONE = 1
CHAIN_APPROX_SIMPLE = 2

CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4


def resize(src, dsize, interpolation=INTER_AREA):
    arr = np.asarray(src)
    dtype = arr.dtype
    pil_mode = "F" if np.issubdtype(dtype, np.floating) and arr.ndim == 2 else None
    img = Image.fromarray(arr.astype(np.float32), mode=pil_mode) if pil_mode else Image.fromarray(arr)
    resample = Image.Resampling.BOX if hasattr(Image, "Resampling") else Image.BOX
    out = np.asarray(img.resize(tuple(dsize), resample=resample))
    return out.astype(dtype, copy=False)


def connectedComponentsWithStats(image, connectivity=8):
    mask = np.asarray(image).astype(bool)
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else None
    labels, num_features = ndimage.label(mask, structure=structure)
    num_labels = num_features + 1

    stats = np.zeros((num_labels, 5), dtype=np.int32)
    centroids = np.zeros((num_labels, 2), dtype=np.float64)
    stats[0, CC_STAT_AREA] = int((labels == 0).sum())

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        stats[label_id] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
            int(len(xs)),
        ]
        centroids[label_id] = [float(xs.mean()), float(ys.mean())]

    return num_labels, labels.astype(np.int32), stats, centroids


def findContours(image, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE):
    mask = np.asarray(image).astype(bool)
    contours = []
    for contour in measure.find_contours(mask.astype(np.uint8), 0.5):
        if contour.shape[0] < 2:
            continue
        xy = np.stack([contour[:, 1], contour[:, 0]], axis=1)
        contours.append(np.round(xy).astype(np.int32)[:, None, :])
    hierarchy = None
    return contours, hierarchy


def addWeighted(src1, alpha, src2, beta, gamma):
    out = np.asarray(src1, dtype=np.float32) * alpha + np.asarray(src2, dtype=np.float32) * beta + gamma
    return np.clip(out, 0, 255).astype(np.asarray(src1).dtype)
