#Section is the main pipeline for the project, mainly AI generated

from __future__ import annotations

from pathlib import Path

import numpy as np

from .feature_toggles import FeatureToggles, color_feature_dim, hog_feature_dim
from .features import combined
from .features import edges as edges_mod
from .features import hog_features, hough_lines, saliency
from .features.color import color_histogram
from .features.shape import shape_pipeline
from .preprocessing import preprocess

def extract_features(
    path: str | Path,
    *,
    toggles: FeatureToggles | None = None,
) -> combined.FeatureBundle:
    """Run the full feature-extraction pipeline on a single image."""
    t = toggles if toggles is not None else FeatureToggles()
    t.validate()

    p = preprocess(path)
    hsv = p["hsv"]
    gray = p["gray"]

    if t.include_color:
        color_hist = color_histogram(hsv)
    else:
        color_hist = np.zeros(color_feature_dim(), dtype=np.float32)

    if t.include_hog:
        hog_vec = hog_features.hog_features(gray)
    else:
        hog_vec = np.zeros(hog_feature_dim(), dtype=np.float32)

    edge_vec, edge_art = edges_mod.build_edge_feature_vector(
        gray,
        include_canny=t.include_edge_canny,
        include_laplacian=t.include_edge_laplacian,
        include_sobel=t.include_edge_sobel,
    )
    edges_bin = edges_mod.edges_for_shape_and_hough(edge_art)

    if t.include_shape:
        _, shape_vec = shape_pipeline(edges_bin, gray.shape[:2])
    else:
        shape_vec = np.zeros(4, dtype=np.float32)

    hough_vec = None
    if t.include_hough:
        hough_vec = hough_lines.hough_line_features(edges_bin)

    sal_vec = None
    if t.include_saliency:
        sal_vec = saliency.spectral_residual_saliency(gray, image_rgb=p["rgb"])

    vec = combined.combine_features(
        color_hist,
        hog_vec,
        edge_vec,
        shape_vec,
        hough_vector=hough_vec,
        saliency_vector=sal_vec,
        toggles=t,
    )

    return combined.FeatureBundle(vector=vec)
