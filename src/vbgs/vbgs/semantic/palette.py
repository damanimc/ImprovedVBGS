"""Distinct colors for semantic class visualization."""

from __future__ import annotations

import colorsys

import numpy as np


def semantic_palette(num_classes: int) -> np.ndarray:
    """Return (num_classes, 3) RGB in [0, 1]."""
    colors = np.zeros((num_classes, 3), dtype=np.float32)
    for i in range(num_classes):
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.92)
        colors[i] = (r, g, b)
    colors[0] = (0.12, 0.12, 0.12)
    return colors


def class_color_rgb(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    labels = np.clip(labels, 0, num_classes - 1)
    return semantic_palette(num_classes)[labels]


def label_from_onehot(onehot: np.ndarray) -> np.ndarray:
    onehot = np.asarray(onehot, dtype=np.float32)
    if onehot.ndim == 1:
        return np.int64(np.argmax(onehot))
    return np.argmax(onehot, axis=-1).astype(np.int64)
