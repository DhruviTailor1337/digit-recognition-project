"""
train.py
---------
Command-line training script for the Handwritten Digit Recognition project.

Usage:
    python src/train.py --epochs 15 --batch-size 128

Outputs:
    models/digit_recognition_model.h5   -> trained model
    reports/figures/training_curves.png -> accuracy / loss curves
    reports/figures/confusion_matrix.png-> confusion matrix on test set
    reports/metrics.json                -> final evaluation metrics
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(__file__))
from data_loader import load_dataset
from model import build_cnn_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
REPORTS_DIR = os.path.join(ROOT, "reports")


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the digit recognition CNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test), source = load_dataset()
    print(f"Dataset source: {source}")

    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)
    print(f"Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")

    model = build_cnn_model(input_shape=x_train.shape[1:])
    model.summary()

    print("Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        verbose=2,
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    plot_training_curves(history, os.path.join(FIGURES_DIR, "training_curves.png"))
    plot_confusion_matrix(y_test, y_pred, os.path.join(FIGURES_DIR, "confusion_matrix.png"))

    model_path = os.path.join(MODELS_DIR, "digit_recognition_model.h5")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    metrics = {
        "dataset_source": source,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()
