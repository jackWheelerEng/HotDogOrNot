#Section mainly uses HW and demo code,  expiremented with houghLinesP

from __future__ import annotations

import numpy as np


def hough_line_features(
    edges_binary: np.ndarray,
    *,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 30,
    min_line_length: int = 20,
    max_line_gap: int = 10,
) -> np.ndarray:
    import cv2

    lines = cv2.HoughLinesP(
        edges_binary,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) == 0:
        return np.zeros(4, dtype=np.float32)

    lengths = []
    for seg in lines[:, 0]:
        x1, y1, x2, y2 = seg
        lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
    lengths_arr = np.array(lengths, dtype=np.float64)
    n = len(lengths_arr)
    return np.array(
        [
            np.log1p(n),
            float(np.mean(lengths_arr)),
            float(np.std(lengths_arr)) if n > 1 else 0.0,
            float(np.max(lengths_arr)),
        ],
        dtype=np.float32,
    )
