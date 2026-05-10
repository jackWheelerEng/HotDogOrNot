"""Shape features from the largest contour.

Pipeline coverage:
    Find largest contour and shape ratio.

Hot dogs are roughly cylindrical — their bounding boxes have a strong
aspect ratio. Extent (contour area / bbox area) and solidity (contour area /
convex hull area) also help separate elongated foods from round ones.
"""

from __future__ import annotations

import numpy as np


def largest_contour(edges: np.ndarray) -> np.ndarray | None:
    """Return the largest external contour in an edge / binary mask."""
    import cv2

    if edges.ndim != 2:
        raise ValueError("edges must be single-channel")
    # demo10_testContour.py — cv2.findContours (RETR_EXTERNAL for largest blob)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def shape_features(
    contour: np.ndarray | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Shape descriptors: ``[aspect_ratio, extent, solidity, normalized_area]``."""
    import cv2

    h, w = image_shape[0], image_shape[1]
    img_area = float(h * w)
    if contour is None or img_area <= 0:
        return np.zeros(4, dtype=np.float32)

    x, y, bw, bh = cv2.boundingRect(contour)
    aspect = float(bw) / float(bh) if bh > 0 else 0.0
    area = float(cv2.contourArea(contour))
    rect_area = float(bw * bh)
    extent = area / rect_area if rect_area > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    norm_area = area / img_area

    return np.array([aspect, extent, solidity, norm_area], dtype=np.float32)


def shape_pipeline(
    edges: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return ``(largest_contour, shape_feature_vector)``."""
    cnt = largest_contour(edges)
    return cnt, shape_features(cnt, image_shape)
