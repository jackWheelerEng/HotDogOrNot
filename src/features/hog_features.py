#Section is built with the suggestion of AI 

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
) -> np.ndarray:
    from skimage.feature import hog

    out = hog(
        image_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
        visualize=False,
    )
    return out.astype(np.float32)
