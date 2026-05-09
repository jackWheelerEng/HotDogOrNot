"""End-to-end feature extraction for a single image.

This is the *spine* of the project — every other module plugs into here. Walk
through it top to bottom and you'll see the pipeline diagram from the README:

    Input image
       ↓ preprocessing.preprocess
    Resize image, convert to HSV and grayscale
       ↓ features.color
    Extract color histogram
       ↓ features.hog_features
    Extract HOG features
       ↓ features.edges
    Run Canny edge detection + edge density
       ↓ features.shape
    Find largest contour and shape ratio
       ↓ features.combined.combine_features
    Combine all features into one vector
"""

from __future__ import annotations

from pathlib import Path

from .features.combined import FeatureBundle


def extract_features(
    path: str | Path,
    *,
    return_visualizations: bool = False,
) -> FeatureBundle:
    """Run the full feature-extraction pipeline on a single image.

    When ``return_visualizations`` is True, intermediate images (HOG render,
    edge map, contour overlay, etc.) are populated on the returned
    ``FeatureBundle`` so ``visualize.py`` can render them.
    """
    raise NotImplementedError
