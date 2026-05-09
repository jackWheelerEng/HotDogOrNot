"""Shared configuration constants for the HotDogOrNot pipeline.

Keeping these in one place makes it easy to tune the pipeline without hunting
through individual feature modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

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

# Random Forest hyperparameters.
RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int | None = None
RF_RANDOM_STATE: int = 42

MODEL_FILENAME: str = "hotdog_rf.joblib"
