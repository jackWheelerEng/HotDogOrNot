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
    """Return the largest external contour in an edge / binary mask.

    Returns ``None`` if no contour is found.
    """
    raise NotImplementedError


def shape_features(contour: np.ndarray | None) -> np.ndarray:
    """Compute shape descriptors for a contour.

    Returns a fixed-length 1D ``float32`` vector of:
        [aspect_ratio, extent, solidity, normalized_area]

    All zeros if ``contour`` is ``None``, so the feature vector length is
    stable across images.
    """
    raise NotImplementedError


def shape_pipeline(
    edges: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Convenience: return ``(largest_contour, shape_feature_vector)``."""
    raise NotImplementedError
