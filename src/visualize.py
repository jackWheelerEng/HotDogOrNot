"""Render image-processing evidence for a single prediction.

Pipeline coverage:
    Show image-processing evidence.

The goal is a single matplotlib figure (or saved PNG) that shows, side by
side, every intermediate step the classifier "saw":

    [original | HSV channels | grayscale]
    [HOG visualization | Canny edges | largest-contour overlay]
    [color histogram bars | shape-feature bars | predicted label + score]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features.combined import FeatureBundle


def render_evidence(
    bundle: FeatureBundle,
    *,
    original_rgb: np.ndarray,
    label: str,
    confidence: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Render the full evidence figure for a prediction.

    If ``save_path`` is provided, the figure is also written to disk (PNG).
    """
    raise NotImplementedError
