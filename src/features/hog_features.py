"""Histogram of Oriented Gradients (HOG) feature.

Pipeline coverage:
    Extract HOG features (captures the elongated cylindrical bun shape).
"""

from __future__ import annotations

import numpy as np

from ..config import (
    HOG_BLOCK_NORM,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
)


def hog_features(
    image_gray: np.ndarray,
    *,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple[int, int] = HOG_CELLS_PER_BLOCK,
    block_norm: str = HOG_BLOCK_NORM,
    return_visualization: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute a HOG descriptor for a grayscale image.

    When ``return_visualization`` is True, also returns a HOG visualization
    image (used by ``visualize.py`` to show the gradient orientations).
    """
    raise NotImplementedError
