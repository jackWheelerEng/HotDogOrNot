"""Edge features.

Pipeline coverage:
    Run Canny edge detection → Calculate edge density.
"""

from __future__ import annotations

import numpy as np

from ..config import CANNY_HIGH, CANNY_LOW


def canny_edges(
    image_gray: np.ndarray,
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
) -> np.ndarray:
    """Run Canny edge detection on a grayscale image.

    Returns a binary ``uint8`` edge map with values in ``{0, 255}``.
    """
    raise NotImplementedError


def edge_density(edges: np.ndarray) -> float:
    """Fraction of pixels that are edge pixels.

    A single scalar in ``[0, 1]``. Useful as a feature *and* a sanity check
    when visualizing the pipeline.
    """
    raise NotImplementedError


def edge_features(image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return ``(edges_image, feature_vector)``.

    The feature vector is currently just ``[edge_density]`` but is shaped as a
    1D array so it concatenates cleanly with the other features.
    """
    raise NotImplementedError
