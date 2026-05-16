"""Train the HotDogOrNot Random Forest on the labeled data directories.

Usage::

    python scripts/train.py
    python scripts/train.py --defaults   # skip prompt; all extraction blocks on
"""
#Section is AI Generated to help bridge my skills for machine learning
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model
from src.config import HOTDOG_DIR, NOT_HOTDOG_DIR
from src.feature_prompt import describe_toggles, prompt_feature_toggles
from src.feature_toggles import FeatureToggles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HotDogOrNot Random Forest")
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Run with every feature block on (no interactive prompt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    toggles = FeatureToggles() if args.defaults else prompt_feature_toggles()
    print(f"Feature set: {describe_toggles(toggles)}\n")

    X, y = model.build_dataset(HOTDOG_DIR, NOT_HOTDOG_DIR, toggles=toggles)
    result = model.train(X, y)
    model.save(result.classifier)

    oob_line = (
        f"  OOB accuracy:   {result.oob_accuracy:.3f}\n"
        if result.oob_accuracy is not None
        else "  OOB accuracy:   (n/a)\n"
    )
    print(
        f"trained on {result.n_train} images, "
        f"held-out {result.n_test} images\n"
        f"  train accuracy: {result.train_accuracy:.3f}  "
        "(often high: RF fits these exact rows)\n"
        f"{oob_line}"
        f"  test  accuracy: {result.test_accuracy:.3f}  "
        "(held-out set — use this + OOB to compare features)"
    )


if __name__ == "__main__":
    main()
