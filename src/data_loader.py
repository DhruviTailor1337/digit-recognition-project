"""
data_loader.py
----------------
Loads the handwritten digit dataset used for training.

Primary source : MNIST (70,000 grayscale images, 28x28, 10 classes).
                 This is the exact same data distributed on Kaggle as the
                 "Digit Recognizer" competition dataset
                 (https://www.kaggle.com/competitions/digit-recognizer)
                 and is fetched here through tf.keras.datasets.mnist,
                 which mirrors the official Kaggle/LeCun MNIST files.

Offline fallback : If no internet connection is available (e.g. while
                 developing inside a sandboxed / offline environment),
                 this module automatically falls back to the
                 scikit-learn "Optical Recognition of Handwritten Digits"
                 dataset (1,797 images, 8x8) and upsamples it to 28x28 so
                 the rest of the pipeline (model, app, predict script)
                 does not need to change.

Usage:
    from data_loader import load_dataset
    (x_train, y_train), (x_test, y_test), source = load_dataset()
"""

import numpy as np


def _load_full_mnist():
    """Try to load the full 70k-image MNIST dataset (Kaggle/LeCun source)."""
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def _load_sklearn_digits_as_fallback():
    """
    Offline fallback dataset: sklearn's bundled handwritten digits dataset.
    Images are upscaled from 8x8 -> 28x28 (bicubic) so shapes match MNIST
    and every downstream component (CNN input shape, Streamlit app,
    predict.py) works unmodified regardless of which source was used.
    """
    from sklearn.datasets import load_digits
    from PIL import Image

    digits = load_digits()
    images = digits.images  # (1797, 8, 8), values 0-16
    labels = digits.target.astype(np.uint8)

    resized = np.zeros((images.shape[0], 28, 28), dtype=np.uint8)
    for i, img in enumerate(images):
        img_uint8 = (img / 16.0 * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).resize((28, 28), Image.BICUBIC)
        resized[i] = np.array(pil_img)

    # simple 80/20 split, shuffled with a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(resized))
    split = int(0.8 * len(resized))
    train_idx, test_idx = idx[:split], idx[split:]

    return (resized[train_idx], labels[train_idx]), (resized[test_idx], labels[test_idx])


def load_dataset(prefer_full_mnist: bool = True):
    """
    Returns (x_train, y_train), (x_test, y_test), source_name

    source_name is one of: "mnist-70k" or "sklearn-digits-1797-fallback"
    """
    if prefer_full_mnist:
        try:
            (x_train, y_train), (x_test, y_test) = _load_full_mnist()
            return (x_train, y_train), (x_test, y_test), "mnist-70k"
        except Exception as e:
            print(f"[data_loader] Could not download full MNIST ({e}).")
            print("[data_loader] Falling back to the offline scikit-learn "
                  "handwritten-digits dataset (1,797 images).")

    (x_train, y_train), (x_test, y_test) = _load_sklearn_digits_as_fallback()
    return (x_train, y_train), (x_test, y_test), "sklearn-digits-1797-fallback"


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), source = load_dataset()
    print(f"Loaded dataset from: {source}")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
