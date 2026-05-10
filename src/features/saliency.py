"""Spectral residual saliency: Demo12 OpenCV API when available, else FFT fallback."""

from __future__ import annotations

import numpy as np


def _features_from_sal_norm(sal_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared (uint8 map, float32 feature vector) from saliency in [0, 1]."""
    sal_u8 = (np.clip(sal_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    mean_s = float(np.mean(sal_norm))
    std_s = float(np.std(sal_norm))
    max_s = float(np.max(sal_norm))
    med = float(np.median(sal_norm))
    high_frac = float(np.mean(sal_norm > med))
    feat = np.array([mean_s, std_s, max_s, high_frac], dtype=np.float32)
    return sal_u8, feat


def _saliency_demo12_opencv(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Demo12_saliency-2.py — StaticSaliencySpectralResidual + computeSaliency (RGB)."""
    import cv2

    sal_mod = getattr(cv2, "saliency", None)
    if sal_mod is None:
        return None
    try:
        saliency = sal_mod.StaticSaliencySpectralResidual_create()
        ok, sal_map = saliency.computeSaliency(image_rgb)
    except cv2.error:
        return None
    if not ok or sal_map is None or sal_map.size == 0:
        return None
    sm = np.asarray(sal_map, dtype=np.float64)
    if sm.ndim == 3:
        sm = np.squeeze(sm)
    smin = float(np.min(sm))
    smax = float(np.max(sm))
    if smax - smin <= 1e-12:
        sal_norm = np.zeros_like(sm, dtype=np.float64)
    else:
        sal_norm = (sm - smin) / (smax - smin)
    return _features_from_sal_norm(sal_norm)


def _saliency_fft_fallback(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Not Demo12 — log-amplitude spectral residual + ifft (no opencv-contrib)."""
    import cv2

    x = gray.astype(np.float64)
    x = x - np.mean(x)
    h, w = x.shape
    f = np.fft.fft2(x)
    amplitude = np.abs(f)
    phase = np.angle(f)
    eps = 1e-8
    log_amplitude = np.log(amplitude + eps)
    smoothed = cv2.blur(log_amplitude.astype(np.float32), (3, 3))
    residual_log = log_amplitude - smoothed
    sal_complex = np.exp(residual_log + 1j * phase)
    saliency = np.abs(np.fft.ifft2(sal_complex)) ** 2
    sigma = max(1.0, min(h, w) * 0.02)
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)

    smin = float(np.min(saliency))
    smax = float(np.max(saliency))
    if smax - smin > 1e-12:
        sal_norm = (saliency - smin) / (smax - smin)
    else:
        sal_norm = np.zeros_like(saliency, dtype=np.float64)
    return _features_from_sal_norm(sal_norm)


def spectral_residual_saliency(
    gray: np.ndarray,
    *,
    image_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(saliency_uint8, feature_vector)`` with 4 float summaries."""
    if image_rgb is not None:
        out = _saliency_demo12_opencv(image_rgb)
        if out is not None:
            return out
    return _saliency_fft_fallback(gray)
