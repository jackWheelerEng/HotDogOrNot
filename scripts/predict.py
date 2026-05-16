"""Classify a single image; print label and confidence to stdout.

Usage::

    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py path/to/image.jpg --defaults
"""
#Section is AI Generated to help bridge my skills for machine learning
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model, pipeline
from src.feature_prompt import describe_toggles, prompt_feature_toggles
from src.feature_toggles import FeatureToggles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hot dog or not?")
    parser.add_argument("image", type=Path, help="Path to an image to classify")
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

    bundle = pipeline.extract_features(args.image, toggles=toggles)
    classifier = model.load()
    label, confidence = model.predict(classifier, bundle.vector)
    print(f"{args.image.name}: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
