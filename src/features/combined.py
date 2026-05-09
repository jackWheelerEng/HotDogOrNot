"""Concatenate every feature into a single vector for the classifier.

Pipeline coverage:
    Combine all features into one vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureBundle:
    """Per-image features kept around so we can both classify and visualize.

    ``vector`` is what feeds the Random Forest; the rest are the intermediate
    artifacts that ``visualize.py`` uses to show the pipeline's evidence.
    """

    vector: np.ndarray
    color_hist: np.ndarray
    hog_vector: np.ndarray
    hog_image: np.ndarray | None
    edges: np.ndarray
    edge_density: float
    contour: np.ndarray | None
    shape_vector: np.ndarray


def combine_features(
    color_hist: np.ndarray,
    hog_vector: np.ndarray,
    edge_feature_vector: np.ndarray,
    shape_vector: np.ndarray,
) -> np.ndarray:
    """Concatenate per-feature vectors into one 1D ``float32`` array."""
    raise NotImplementedError
