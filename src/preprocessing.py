#Section mainly uses HW code with some extra OpenCV functions 

from __future__ import annotations

from pathlib import Path

from .config import IMAGE_SIZE


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR ``uint8`` array."""
    import cv2

    p = str(path)
    # HW1/HW2 — cv2.imread (BGR)
    data = cv2.imread(p, cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return data


def resize_image_bgr(
    image_bgr: np.ndarray,
    size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize a BGR image. ``size`` is ``(width, height)``."""
    import cv2

    w, h = size
    # Demo3_PixelBasics.ipynb — cv2.resize
    return cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)


def preprocess(path: str | Path) -> dict[str, np.ndarray]:
    """Run load → resize on BGR, then grayscale and HSV like HW1/HW2.

    Returns keys ``bgr``, ``rgb``, ``hsv``, ``gray``.
    """
    import cv2

    bgr = resize_image_bgr(load_image_bgr(path))
    # HW1/HW2 — BGR2GRAY
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Not HW — BGR2HSV (for HSV histogram)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return {"bgr": bgr, "rgb": rgb, "hsv": hsv, "gray": gray}
