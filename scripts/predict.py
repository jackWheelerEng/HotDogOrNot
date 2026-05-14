"""Classify a single image; print label and confidence to stdout.

Usage::

    python scripts/predict.py path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hot dog or not?")
    parser.add_argument("image", type=Path, help="Path to an image to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = pipeline.extract_features(args.image)
    classifier = model.load()
    label, confidence = model.predict(classifier, bundle.vector)
    print(f"{args.image.name}: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
