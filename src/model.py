"""Random Forest classifier — train, save, load, predict.

Pipeline coverage:
    Train random forest classifier → Predict hot dog / not hot dog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from . import pipeline
from .config import (
    CLASS_NAMES,
    MODEL_FILENAME,
    MODELS_DIR,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
)

_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


def _collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(p)
    return paths


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
    paths_labels: list[tuple[Path, int]] = [
        *[(p, 0) for p in _collect_images(not_hotdog_dir)],
        *[(p, 1) for p in _collect_images(hotdog_dir)],
    ]
    if not paths_labels:
        raise ValueError(
            "No labeled images found. Add files with extensions "
            f"{sorted(_IMAGE_SUFFIXES)} under {not_hotdog_dir!s} and {hotdog_dir!s}."
        )

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for path, label in paths_labels:
        bundle = pipeline.extract_features(path)
        rows.append(bundle.vector)
        labels.append(label)

    X = np.stack(rows, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
) -> TrainResult:
    """Fit a Random Forest with a held-out split and report accuracy."""
    unique, counts = np.unique(y, return_counts=True)
    stratify = y if unique.size > 1 and int(counts.min()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RF_RANDOM_STATE,
        stratify=stratify,
    )
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return TrainResult(
        classifier=clf,
        train_accuracy=float(clf.score(X_train, y_train)),
        test_accuracy=float(clf.score(X_test, y_test)),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
    )


def save(classifier: RandomForestClassifier, path: Path | None = None) -> Path:
    """Persist the trained classifier to ``models/`` (joblib)."""
    dest = Path(path) if path is not None else MODELS_DIR / MODEL_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, dest)
    return dest


def load(path: Path | None = None) -> RandomForestClassifier:
    """Load a previously trained classifier."""
    src = Path(path) if path is not None else MODELS_DIR / MODEL_FILENAME
    if not src.is_file():
        raise FileNotFoundError(f"No trained model at {src}")
    clf = joblib.load(src)
    if not isinstance(clf, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier at {src}, got {type(clf).__name__}")
    return clf


def predict(
    classifier: RandomForestClassifier,
    feature_vector: np.ndarray,
) -> tuple[str, float]:
    """Classify a single feature vector.

    Returns ``(label, confidence)`` where ``label`` is one of
    :data:`config.CLASS_NAMES` and ``confidence`` is the predicted-class
    probability in ``[0, 1]``.
    """
    v = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    proba = classifier.predict_proba(v)[0]
    col = int(np.argmax(proba))
    class_id = int(classifier.classes_[col])
    return CLASS_NAMES[class_id], float(proba[col])


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
