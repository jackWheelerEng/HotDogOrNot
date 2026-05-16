# HotDogOrNot

A project inspired by the *Silicon Valley* "SeeFood" bit: given an image, decide whether it shows a hot dog or not. It uses classical computer-vision features and a Random Forest classifier. 

## Pipeline

```
Input image
   ↓
Resize image
   ↓
Convert to HSV and grayscale
   ↓
Color histogram                  ─┐
Extract HOG features              ├─→  Combine into one feature vector
Edge detection                    │
Hough line transform              │
Saliency                          │
Shape (solidity and extent)       ┘
   ↓
Train Random Forest classifier
   ↓
Predict hot dog / not hot dog
```


## Project layout

```
HotDogOrNot/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── hotdog/         # labeled hot dog images
│       └── not_hotdog/     # labeled non-hot-dog images
├── models/                 # trained classifier (.joblib)
├── scripts/
│   ├── train.py            # optional feature prompts; persists the model
│   └── predict.py          # classify one image; print label & confidence
└── src/
    ├── config.py
    ├── feature_prompt.py   # CLI prompts for enabling / disabling feature blocks
    ├── feature_toggles.py
    ├── preprocessing.py
    ├── pipeline.py         # feature extraction entry point
    ├── model.py            # dataset build, RF train / save / load / predict
    └── features/
        ├── color.py
        ├── hog_features.py
        ├── edges.py        
        ├── shape.py
        ├── hough_lines.py
        ├── saliency.py
        └── combined.py     # stack enabled blocks into one vector
```

## Setup

Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the Project
```bash
python scripts/train.py
#Alternativly, use the default setting to use all features
python scripts/train.py --defaults 
```

**Dependencies:** NumPy, OpenCV (contrib build for optional saliency API), scikit-image (HOG), scikit-learn (Random Forest), joblib (model persistence).
