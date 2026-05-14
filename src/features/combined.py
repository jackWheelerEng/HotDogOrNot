"""Concatenate every feature into one vector for the classifier.

Pipeline coverage:
    Combine all features into one vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import INCLUDE_HOUGH, INCLUDE_SALIENCY


@dataclass
class FeatureBundle:
    """Per-image features: ``vector`` for the classifier; other fields are intermediates."""


    vector: np.ndarray
    color_hist: np.ndarray
    hog_vector: np.ndarray
    hog_image: np.ndarray | None
    edges: np.ndarray
    edge_density: float
    contour: np.ndarray | None
    shape_vector: np.ndarray
    gray_smoothed: np.ndarray | None = None
    laplacian_mag: np.ndarray | None = None
    laplacian_binary: np.ndarray | None = None
    saliency_map: np.ndarray | None = None


def combine_features(
    color_hist: np.ndarray,
    hog_vector: np.ndarray,
    edge_vector: np.ndarray,
    shape_vector: np.ndarray,
    *,
    hough_vector: np.ndarray | None = None,
    saliency_vector: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate per-feature vectors into one 1D ``float32`` array.

    Hough and saliency blocks are included when the matching ``INCLUDE_*`` flag
    in :mod:`config` is True and the vector argument is not ``None``.
    """
    blocks: list[np.ndarray] = [
        color_hist.astype(np.float32, copy=False),
        hog_vector.astype(np.float32, copy=False),
        edge_vector.astype(np.float32, copy=False),
        shape_vector.astype(np.float32, copy=False),
    ]
    if INCLUDE_HOUGH and hough_vector is not None:
        blocks.append(hough_vector.astype(np.float32, copy=False))
    if INCLUDE_SALIENCY and saliency_vector is not None:
        blocks.append(saliency_vector.astype(np.float32, copy=False))

    return np.concatenate(blocks).astype(np.float32)
