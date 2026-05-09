# HotDogOrNot

A classical computer-vision take on the *Silicon Valley* "SeeFood" gag: given an
image, decide whether it contains a hot dog or not. No deep learning required —
the goal is to lean on hand-crafted image features and a Random Forest, and to
*show the work* with intermediate visualizations.

## Pipeline

```
Input image
   ↓
Resize image
   ↓
Convert to HSV and grayscale
   ↓
Extract color histogram          ─┐
Extract HOG features              ├─→  Combine into one feature vector
Run Canny edge detection          │
Calculate edge density            │
Find largest contour + shape ratio┘
   ↓
Train Random Forest classifier
   ↓
Predict hot dog / not hot dog
   ↓
Show image-processing evidence
```

Each step lives in its own module under `src/` so you can iterate on one piece
at a time (e.g. swap Canny for Laplacian, add saliency, try a Hough transform).

## Project layout

```
HotDogOrNot/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── hotdog/         # put hot dog images here
│   │   └── not_hotdog/     # put non-hot-dog images here
│   └── processed/          # cached feature vectors (optional)
├── models/                 # trained classifier artifacts (.joblib)
├── outputs/                # saved evidence figures
├── scripts/
│   ├── train.py            # train + persist Random Forest
│   └── predict.py          # classify a single image + show evidence
└── src/
    ├── config.py           # shared constants (image size, HOG params, ...)
    ├── preprocessing.py    # resize / HSV / grayscale
    ├── features/
    │   ├── color.py        # HSV color histogram
    │   ├── hog_features.py # HOG descriptor
    │   ├── edges.py        # Canny + edge density
    │   ├── shape.py        # largest contour + shape ratio
    │   └── combined.py     # concatenate everything into one vector
    ├── pipeline.py         # end-to-end feature extraction for one image
    ├── model.py            # Random Forest train / save / load / predict
    └── visualize.py        # render image-processing evidence
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (planned)

```bash
# 1. drop images into data/raw/hotdog/ and data/raw/not_hotdog/
# 2. train the classifier
python scripts/train.py

# 3. classify a new image and see the evidence
python scripts/predict.py path/to/image.jpg
```

## Possible extensions

These are intentionally left out of the core pipeline but slot in cleanly:

- Saliency map (`cv2.saliency`) as an extra feature channel or visualization.
- Hough transforms for cylindrical-bun detection.
- FFT / DCT frequency features.
- Laplacian-of-Gaussian as an alternative to Canny.
- Simple segmentation mask (GrabCut, Otsu on saturation) to gate other features.
