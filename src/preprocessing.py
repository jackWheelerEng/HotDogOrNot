"""Preprocessing: load → resize → HSV / grayscale.

Pipeline coverage:
    Input image → Resize image → Convert to HSV and grayscale
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import IMAGE_SIZE


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB ``uint8`` array.

    OpenCV reads as BGR by default; this function returns RGB so downstream
    code (matplotlib, scikit-image) sees what humans expect.
    """
    raise NotImplementedError


def resize_image(
    image_rgb: np.ndarray,
    size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize an RGB image to the canonical pipeline size."""
    raise NotImplementedError


def to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSV color space."""
    raise NotImplementedError


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale."""
    raise NotImplementedError


def preprocess(path: str | Path) -> dict[str, np.ndarray]:
    """Run load → resize → HSV/grayscale and return all three views.

    Returns a dict with keys ``rgb``, ``hsv``, ``gray`` so feature modules can
    pick whichever representation they need without re-converting.
    """
    raise NotImplementedError
