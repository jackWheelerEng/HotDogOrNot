"""Edge features: Gaussian blur, Sobel, Laplacian, Canny."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from ..config import (
    CANNY_HIGH,
    CANNY_LOW,
    GAUSSIAN_HW_KERNEL_SIZE,
    INCLUDE_EDGE_CANNY,
    INCLUDE_EDGE_LAPLACIAN,
    INCLUDE_EDGE_SOBEL,
    LAPLACIAN_EDGE_PERCENTILE,
    USE_GAUSSIAN_BEFORE_EDGES,
    USE_HW3_GAUSSIAN_FILTER2D,
)


class EdgeArtifacts(TypedDict, total=False):
    gray_smoothed: np.ndarray
    canny: np.ndarray
    laplacian_mag: np.ndarray
    laplacian_binary: np.ndarray


def gaussian_kernel_hw3(kernel_size: int = GAUSSIAN_HW_KERNEL_SIZE) -> np.ndarray:
    """2D Gaussian kernel normalized to sum 1."""
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    # HW3 — Gaussian kernel
    g_ker = np.zeros((k, k), dtype=np.float64)
    center = k // 2
    sigma_kernel = k / 5.0
    for i in range(k):
        for j in range(k):
            x_dist = i - center
            y_dist = j - center
            g_ker[i, j] = np.exp(-(x_dist**2 + y_dist**2) / (2 * sigma_kernel**2))
    g_ker /= np.sum(g_ker)
    return g_ker.astype(np.float32)


def gaussian_smooth_gray_hw3(image_gray: np.ndarray) -> np.ndarray:
    """Gaussian blur via filter2D; output uint8 clipped to [0, 255]."""
    import cv2

    g_ker = gaussian_kernel_hw3()
    # HW3 — filter2D (Gaussian blur)
    blurred = cv2.filter2D(image_gray.astype(np.float32), cv2.CV_32F, g_ker)
    return np.clip(blurred, 0, 255).astype(np.uint8)


_DEMO56_GAUSSIAN_KERNEL = np.array(
    [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ],
    dtype=np.float32,
)
_DEMO56_GAUSSIAN_KERNEL /= float(np.sum(_DEMO56_GAUSSIAN_KERNEL))


def gaussian_smooth_gray_demo56(image_gray: np.ndarray) -> np.ndarray:
    """5×5 discrete Gaussian from course demos; output uint8 clipped."""
    import cv2

    # Demo5_2dFilters-6.py / Demo6 — Gaussian filter + filter2D
    blurred = cv2.filter2D(image_gray.astype(np.float32), cv2.CV_32F, _DEMO56_GAUSSIAN_KERNEL)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def gaussian_smooth_gray(image_gray: np.ndarray) -> np.ndarray:
    """Gaussian blur: HW3 loop kernel or Demo5/Demo6 discrete kernel."""
    if USE_HW3_GAUSSIAN_FILTER2D:
        return gaussian_smooth_gray_hw3(image_gray)
    return gaussian_smooth_gray_demo56(image_gray)


def prepare_gray_for_edges(image_gray: np.ndarray) -> np.ndarray:
    """Return grayscale optionally smoothed before edge operators."""
    if USE_GAUSSIAN_BEFORE_EDGES:
        return gaussian_smooth_gray(image_gray)
    return image_gray


def laplacian_kernel_demo56() -> np.ndarray:
    """3×3 Laplacian with diagonals (l_kern2 in demos)."""
    # Demo5and6_2dFiltering.ipynb / Demo6 — Laplacian (l_kern2)
    return np.array(
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
        dtype=np.float32,
    )


def laplacian_response_demo56(image_gray: np.ndarray) -> np.ndarray:
    """Discrete Laplacian via filter2D."""
    import cv2

    l_kern = laplacian_kernel_demo56()
    # Demo5/Demo6 — filter2D (Laplacian)
    return cv2.filter2D(image_gray.astype(np.float32), cv2.CV_32F, l_kern)


def sobel_gradient_magnitude_demo56(image_gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude; float32."""
    import cv2

    # Demo5_2dFilters-6.py / Demo6 — Sobel + filter2D
    sobel_vert = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sobel_horiz = sobel_vert.T
    d_h = cv2.filter2D(image_gray.astype(np.float32), cv2.CV_32F, sobel_horiz)
    d_v = cv2.filter2D(image_gray.astype(np.float32), cv2.CV_32F, sobel_vert)
    grad = np.sqrt(np.square(d_h) + np.square(d_v))
    gmax = float(np.max(grad)) if grad.size else 0.0
    if gmax > 0:
        grad = grad * (255.0 / gmax)
    return grad.astype(np.float32)


def canny_edges(
    image_gray: np.ndarray,
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
) -> np.ndarray:
    """Run Canny edge detection on a grayscale image.

    Returns a binary ``uint8`` edge map with values in ``{0, 255}``.
    """
    import cv2

    # Not HW — Canny edge detection
    return cv2.Canny(image_gray, low, high)


def laplacian_magnitude(image_gray: np.ndarray) -> np.ndarray:
    """|Laplacian response|, float32."""
    return np.abs(laplacian_response_demo56(image_gray)).astype(np.float32)


def edge_density(edges: np.ndarray) -> float:
    """Fraction of pixels that are edge pixels (values > 127 treated as on)."""
    if edges.size == 0:
        return 0.0
    return float(np.mean(edges > 127))


def laplacian_binary_edges(
    lap_mag: np.ndarray,
    percentile: float = LAPLACIAN_EDGE_PERCENTILE,
) -> np.ndarray:
    """Binary edge map from percentile threshold on Laplacian magnitude."""
    t = float(np.percentile(lap_mag, percentile))
    out = (lap_mag >= t).astype(np.uint8) * 255
    return out


def laplacian_scalar_features(lap_mag: np.ndarray) -> np.ndarray:
    """Scalar summaries of Laplacian magnitude (not HW)."""
    flat = lap_mag.ravel()
    mean_abs = float(np.mean(flat))
    std_abs = float(np.std(flat))
    # "Edge density" if we threshold at mean + 1 std (simple, stable).
    thr = mean_abs + std_abs
    dens = float(np.mean(flat >= thr)) if flat.size else 0.0
    return np.array([mean_abs, std_abs, dens], dtype=np.float32)


def sobel_scalar_features(grad_mag: np.ndarray) -> np.ndarray:
    """Scalar summaries of Sobel magnitude (project)."""
    flat = grad_mag.ravel()
    mean_g = float(np.mean(flat))
    std_g = float(np.std(flat))
    thr = mean_g + std_g
    dens = float(np.mean(flat >= thr)) if flat.size else 0.0
    return np.array([mean_g, std_g, dens], dtype=np.float32)


def build_edge_feature_vector(image_gray: np.ndarray) -> tuple[np.ndarray, EdgeArtifacts]:
    """Compute concatenated edge-related features plus visualization artifacts.

    Respects ``INCLUDE_EDGE_*`` flags in config.
    """
    gray_s = prepare_gray_for_edges(image_gray)
    lap_mag = laplacian_magnitude(gray_s)
    lap_bin = laplacian_binary_edges(lap_mag)
    canny = canny_edges(gray_s)
    grad_mag = sobel_gradient_magnitude_demo56(gray_s) if INCLUDE_EDGE_SOBEL else None

    parts: list[np.ndarray] = []
    if INCLUDE_EDGE_CANNY:
        parts.append(np.array([edge_density(canny)], dtype=np.float32))
    if INCLUDE_EDGE_LAPLACIAN:
        parts.append(laplacian_scalar_features(lap_mag))
    if INCLUDE_EDGE_SOBEL and grad_mag is not None:
        parts.append(sobel_scalar_features(grad_mag))

    if not parts:
        vec = np.zeros(0, dtype=np.float32)
    else:
        vec = np.concatenate(parts).astype(np.float32)

    art: EdgeArtifacts = {
        "gray_smoothed": gray_s,
        "canny": canny,
        "laplacian_mag": lap_mag,
        "laplacian_binary": lap_bin,
    }
    return vec, art


def edges_for_shape_and_hough(artifacts: EdgeArtifacts) -> np.ndarray:
    """Pick the binary edge map used for contours + ``HoughLinesP``."""
    from ..config import SHAPE_EDGE_SOURCE

    if SHAPE_EDGE_SOURCE == "laplacian":
        return artifacts["laplacian_binary"]
    return artifacts["canny"]
