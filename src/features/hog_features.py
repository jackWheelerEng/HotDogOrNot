"""HOG feature (not HW)."""

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
    """Compute a HOG descriptor for a grayscale image."""
    # Not HW — HOG
    from skimage.feature import hog

    out = hog(
        image_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
        visualize=return_visualization,
    )
    if return_visualization:
        feat, viz = out
        return feat.astype(np.float32), viz
    return out.astype(np.float32)
