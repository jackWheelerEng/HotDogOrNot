"""Color histogram feature.

Pipeline coverage:
    Extract color histogram (computed over HSV — sausages and buns sit in a
    pretty narrow hue/saturation band, which is the whole point).
"""

from __future__ import annotations

import numpy as np

from ..config import COLOR_HIST_BINS


def color_histogram(
    image_hsv: np.ndarray,
    bins: tuple[int, int, int] = COLOR_HIST_BINS,
) -> np.ndarray:
    """Compute a flattened, L1-normalized 3D HSV histogram.

    Parameters
    ----------
    image_hsv:
        ``(H, W, 3)`` uint8 image in HSV color space.
    bins:
        Number of histogram bins per (H, S, V) channel.

    Returns
    -------
    1D ``float32`` feature vector of length ``prod(bins)``.
    """
    raise NotImplementedError
