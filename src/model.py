"""Random Forest classifier — train, save, load, predict.

Pipeline coverage:
    Train random forest classifier → Predict hot dog / not hot dog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import CLASS_NAMES, MODEL_FILENAME, MODELS_DIR


@dataclass
class TrainResult:
    """Summary of a training run, suitable for printing or logging."""

    classifier: RandomForestClassifier
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int


def build_dataset(
    hotdog_dir: Path,
    not_hotdog_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the labeled image directories and build ``(X, y)``.

    Each image is run through ``pipeline.extract_features`` and its
    ``FeatureBundle.vector`` is stacked into ``X``.
    """
    raise NotImplementedError


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
) -> TrainResult:
    """Fit a Random Forest with a held-out split and report accuracy."""
    raise NotImplementedError


def save(classifier: RandomForestClassifier, path: Path | None = None) -> Path:
    """Persist the trained classifier to ``models/`` (joblib)."""
    raise NotImplementedError


def load(path: Path | None = None) -> RandomForestClassifier:
    """Load a previously trained classifier."""
    raise NotImplementedError


def predict(
    classifier: RandomForestClassifier,
    feature_vector: np.ndarray,
) -> tuple[str, float]:
    """Classify a single feature vector.

    Returns ``(label, confidence)`` where ``label`` is one of
    :data:`config.CLASS_NAMES` and ``confidence`` is the predicted-class
    probability in ``[0, 1]``.
    """
    raise NotImplementedError


__all__ = [
    "CLASS_NAMES",
    "MODEL_FILENAME",
    "MODELS_DIR",
    "TrainResult",
    "build_dataset",
    "train",
    "save",
    "load",
    "predict",
]
