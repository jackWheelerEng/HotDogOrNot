#Prompting the user to toggle features on and off and feeds from feature_toggles.py

from __future__ import annotations

from .feature_toggles import FeatureToggles


def _parse_yes_no(raw: str, default: bool) -> bool:
    s = raw.strip().lower()
    if not s:
        return default
    if s in ("y", "yes", "1", "t", "true"):
        return True
    if s in ("n", "no", "0", "f", "false"):
        return False
    return default


def prompt_feature_toggles(defaults: FeatureToggles | None = None) -> FeatureToggles:
    """Ask y/n for each feature block; blank line keeps the shown default."""
    d = defaults if defaults is not None else FeatureToggles()
    print("\n=== Feature blocks for this run ===")
    print("y = on, n = off, Enter = keep current default.\n")

    def ask(label: str, cur: bool) -> bool:
        line = input(f"  {label}: ")
        return _parse_yes_no(line, cur)

    t = FeatureToggles(
        include_color=ask("calcHist", d.include_color),
        include_hog=ask("hog", d.include_hog),
        include_edge_canny=ask("Canny", d.include_edge_canny),
        include_edge_laplacian=ask("Laplacian", d.include_edge_laplacian),
        include_edge_sobel=ask("Sobel", d.include_edge_sobel),
        include_shape=ask("findContours", d.include_shape),
        include_hough=ask("HoughLinesP", d.include_hough),
        include_saliency=ask("spectral_residual", d.include_saliency),
    )
    t.validate()
    print()
    return t


def describe_toggles(t: FeatureToggles) -> str:
    """One-line summary for logging."""
    parts: list[str] = []
    if t.include_color:
        parts.append("calcHist")
    if t.include_hog:
        parts.append("hog")
    if t.include_edge_canny:
        parts.append("Canny")
    if t.include_edge_laplacian:
        parts.append("Laplacian")
    if t.include_edge_sobel:
        parts.append("Sobel")
    if t.include_shape:
        parts.append("findContours")
    if t.include_hough:
        parts.append("HoughLinesP")
    if t.include_saliency:
        parts.append("spectral_residual")
    return ", ".join(parts) if parts else "(none)"
