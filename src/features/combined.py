# Section uses AI to help with using float32 for the vector

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..feature_toggles import FeatureToggles


@dataclass
class FeatureBundle:
    vector: np.ndarray

def combine_features(
    color_hist: np.ndarray,
    hog_vector: np.ndarray,
    edge_vector: np.ndarray,
    shape_vector: np.ndarray,
    *,
    hough_vector: np.ndarray | None = None,
    saliency_vector: np.ndarray | None = None,
    toggles: FeatureToggles | None = None,
) -> np.ndarray:
    t = toggles if toggles is not None else FeatureToggles()
    blocks: list[np.ndarray] = []
    if t.include_color:
        blocks.append(color_hist.astype(np.float32, copy=False))
    if t.include_hog:
        blocks.append(hog_vector.astype(np.float32, copy=False))
    if t.any_edge_block():
        blocks.append(edge_vector.astype(np.float32, copy=False))
    if t.include_shape:
        blocks.append(shape_vector.astype(np.float32, copy=False))
    if t.include_hough and hough_vector is not None:
        blocks.append(hough_vector.astype(np.float32, copy=False))
    if t.include_saliency and saliency_vector is not None:
        blocks.append(saliency_vector.astype(np.float32, copy=False))

    if not blocks:
        raise ValueError("No feature blocks selected; check FeatureToggles.")
    return np.concatenate(blocks).astype(np.float32)
