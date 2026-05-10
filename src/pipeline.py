"""End-to-end feature extraction for a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import INCLUDE_EDGE_CANNY, INCLUDE_HOUGH, INCLUDE_SALIENCY
from .features import combined
from .features import edges as edges_mod
from .features import hog_features, hough_lines, saliency
from .features.color import color_histogram
from .features.shape import shape_pipeline
from .preprocessing import preprocess

# HW1/HW2 — imread, BGR2GRAY; Demo3_PixelBasics — resize
# HW3 — Gaussian (loop kernel); Demo5/Demo6 — Sobel, Laplacian, discrete Gaussian
# Demo12_saliency-2.py — spectral residual (OpenCV); demo10_testContour.py — findContours
# Not from demos — HOG, HoughLinesP, Canny, HSV histogram


def extract_features(
    path: str | Path,
    *,
    return_visualizations: bool = False,
) -> combined.FeatureBundle:
    """Run the full feature-extraction pipeline on a single image."""
    p = preprocess(path)
    hsv = p["hsv"]
    gray = p["gray"]

    color_hist = color_histogram(hsv)

    if return_visualizations:
        hog_vec, hog_img = hog_features.hog_features(
            gray,
            return_visualization=True,
        )
        assert isinstance(hog_img, np.ndarray)
    else:
        hog_out = hog_features.hog_features(gray, return_visualization=False)
        hog_vec = hog_out
        hog_img = None

    edge_vec, edge_art = edges_mod.build_edge_feature_vector(gray)
    edges_bin = edges_mod.edges_for_shape_and_hough(edge_art)
    contour, shape_vec = shape_pipeline(edges_bin, gray.shape[:2])

    hough_vec = None
    if INCLUDE_HOUGH:
        hough_vec = hough_lines.hough_line_features(edges_bin)

    sal_vec = None
    sal_map = None
    if INCLUDE_SALIENCY:
        sal_map, sal_vec = saliency.spectral_residual_saliency(gray, image_rgb=p["rgb"])

    vec = combined.combine_features(
        color_hist,
        hog_vec,
        edge_vec,
        shape_vec,
        hough_vector=hough_vec,
        saliency_vector=sal_vec,
    )

    canny_map = edge_art["canny"]
    if INCLUDE_EDGE_CANNY:
        ed = edges_mod.edge_density(canny_map)
    else:
        ed = 0.0

    return combined.FeatureBundle(
        vector=vec,
        color_hist=color_hist,
        hog_vector=hog_vec,
        hog_image=hog_img,
        edges=canny_map,
        edge_density=ed,
        contour=contour,
        shape_vector=shape_vec,
        gray_smoothed=edge_art.get("gray_smoothed"),
        laplacian_mag=edge_art.get("laplacian_mag"),
        laplacian_binary=edge_art.get("laplacian_binary"),
        saliency_map=sal_map,
    )
