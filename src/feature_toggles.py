#Section to improve testability of the project by toggling features on and off

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    COLOR_HIST_BINS,
    HOG_BLOCK_NORM,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
)


def color_feature_dim() -> int:
    h_b, s_b, v_b = COLOR_HIST_BINS
    return int(h_b * s_b * v_b)


def hog_feature_dim() -> int:
    """Length of the HOG vector for ``IMAGE_SIZE`` and current HOG config."""
    from skimage.feature import hog

    w, h = IMAGE_SIZE
    gray = np.zeros((h, w), dtype=np.uint8)
    vec = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    return int(np.asarray(vec, dtype=np.float32).size)


@dataclass
class FeatureToggles:
    """Which feature blocks are computed and concatenated for this run."""

    include_color: bool = True
    include_hog: bool = True
    include_edge_canny: bool = True
    include_edge_laplacian: bool = True
    include_edge_sobel: bool = True
    include_shape: bool = True
    include_hough: bool = True
    include_saliency: bool = True

    def any_edge_block(self) -> bool:
        return (
            self.include_edge_canny
            or self.include_edge_laplacian
            or self.include_edge_sobel
        )

    def validate(self) -> None:
        if not (
            self.include_color
            or self.include_hog
            or self.any_edge_block()
            or self.include_shape
            or self.include_hough
            or self.include_saliency
        ):
            raise ValueError(
                "At least one feature block must be enabled (color, HOG, edges, "
                "shape, Hough, or saliency)."
            )
