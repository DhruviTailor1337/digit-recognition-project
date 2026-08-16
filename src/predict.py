"""
predict.py
-----------
Command-line inference script. Loads the trained model and predicts the
digit contained in a single image file.

Usage:
    python src/predict.py --image path/to/digit.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "digit_recognition_model.h5")


def preprocess_image(image_path: str) -> np.ndarray:
    """Loads an image, converts to grayscale, resizes to 28x28,
    inverts colors if needed (model expects white digit on black
    background, like MNIST), and normalizes to [0, 1]."""
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32")

    # If the image looks like a dark digit on a light background
    # (typical of a photo/scan), invert it to match MNIST's
    # light digit on dark background convention.
    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict(image_path: str, model_path: str = DEFAULT_MODEL_PATH):
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return predicted_digit, confidence, probs


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image")
    parser.add_argument("--image", required=True, help="Path to the digit image")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model file")
    args = parser.parse_args()

    digit, confidence, probs = predict(args.image, args.model)
    print(f"Predicted digit : {digit}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print("Class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {i}: {p * 100:5.2f}%")


if __name__ == "__main__":
    main()
