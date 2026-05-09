"""Train the HotDogOrNot Random Forest on the labeled data directories.

Usage::

    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model
from src.config import HOTDOG_DIR, NOT_HOTDOG_DIR


def main() -> None:
    X, y = model.build_dataset(HOTDOG_DIR, NOT_HOTDOG_DIR)
    result = model.train(X, y)
    model.save(result.classifier)

    print(
        f"trained on {result.n_train} images, "
        f"held-out {result.n_test} images\n"
        f"  train accuracy: {result.train_accuracy:.3f}\n"
        f"  test  accuracy: {result.test_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
