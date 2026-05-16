from __future__ import annotations

import numpy as np

from ..config import COLOR_HIST_BINS


def color_histogram(
    image_hsv: np.ndarray,
    bins: tuple[int, int, int] = COLOR_HIST_BINS,
) -> np.ndarray:
    import cv2

    h_b, s_b, v_b = bins
    # New technique from machine learning (AI suggested)
    hist = cv2.calcHist(
        [image_hsv],
        [0, 1, 2],
        None,
        [h_b, s_b, v_b],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)
