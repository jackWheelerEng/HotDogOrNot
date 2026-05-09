"""Classify a single image and show the image-processing evidence.

Usage::

    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py path/to/image.jpg --save outputs/evidence.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model, pipeline, visualize
from src.preprocessing import load_image, resize_image
from src.config import IMAGE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hot dog or not?")
    parser.add_argument("image", type=Path, help="Path to an image to classify")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the evidence figure (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive matplotlib window (useful for headless runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = pipeline.extract_features(args.image, return_visualizations=True)
    classifier = model.load()
    label, confidence = model.predict(classifier, bundle.vector)

    original = resize_image(load_image(args.image), IMAGE_SIZE)
    visualize.render_evidence(
        bundle,
        original_rgb=original,
        label=label,
        confidence=confidence,
        save_path=args.save,
        show=not args.no_show,
    )

    print(f"{args.image.name}: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
