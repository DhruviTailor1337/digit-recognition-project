"""
model.py
---------
Defines the CNN architecture used for handwritten digit recognition.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds and compiles a compact CNN suitable for MNIST-style digit
    classification.

    Architecture:
        Conv2D(32) -> ReLU -> MaxPool
        Conv2D(64) -> ReLU -> MaxPool
        Flatten
        Dense(128) -> ReLU -> Dropout(0.5)
        Dense(num_classes) -> Softmax
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="digit_recognition_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_cnn_model()
    m.summary()
