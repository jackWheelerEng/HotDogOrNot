#Section of project parameters, mainly AI generated 
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
MODELS_DIR: Path = PROJECT_ROOT / "models"

HOTDOG_DIR: Path = RAW_DIR / "hotdog"
NOT_HOTDOG_DIR: Path = RAW_DIR / "not_hotdog"

CLASS_NAMES: tuple[str, str] = ("not_hotdog", "hotdog")

IMAGE_SIZE: tuple[int, int] = (128, 128)

# HSV color histogram: bins per channel.
COLOR_HIST_BINS: tuple[int, int, int] = (16, 16, 16)

# HOG parameters (skimage.feature.hog).
HOG_ORIENTATIONS: int = 9
HOG_PIXELS_PER_CELL: tuple[int, int] = (16, 16)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
HOG_BLOCK_NORM: str = "L2-Hys"

# Canny edge detection thresholds.
CANNY_LOW: int = 80
CANNY_HIGH: int = 160

# Gaussian blur before edge operators (classic denoise before Laplacian / Canny).
USE_GAUSSIAN_BEFORE_EDGES: bool = True
# HW3 — large loop-built Gaussian; else Demo5/Demo6 — 5×5 discrete Gaussian + filter2D
USE_HW3_GAUSSIAN_FILTER2D: bool = True
# HW3 — Gaussian kernel size
GAUSSIAN_HW_KERNEL_SIZE: int = 45

# Laplacian: global percentile threshold on |∇²I| → binary map for viz / shape / Hough.
LAPLACIAN_EDGE_PERCENTILE: float = 92.0

# Which binarized edge map feeds contour finding + Hough (``"canny"`` or ``"laplacian"``).
SHAPE_EDGE_SOURCE: str = "canny"

# Random Forest hyperparameters.
RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int | None = None
RF_RANDOM_STATE: int = 42

MODEL_FILENAME: str = "hotdog_rf.joblib"
