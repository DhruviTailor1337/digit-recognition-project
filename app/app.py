"""
app.py
-------
Streamlit web application for the Handwritten Digit Recognition project.

Lets a user either:
  1. Draw a digit on an in-browser canvas, or
  2. Upload an image of a handwritten digit,

and returns the model's predicted digit with a confidence chart.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "digit_recognition_model.h5")

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="centered")


@st.cache_resource
def load_trained_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L").resize((28, 28))
    arr = np.array(img).astype("float32")
    if arr.mean() > 127:
        arr = 255 - arr
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1), arr


def main():
    st.title("✍️ Handwritten Digit Recognition")
    st.caption("CNN trained on the MNIST handwritten digit dataset")

    st.write(
        "Upload an image of a single handwritten digit (0-9), or use the "
        "drawing canvas below, and the model will predict which digit it is."
    )

    tab_draw, tab_upload = st.tabs(["🖊️ Draw a digit", "📁 Upload an image"])
    image_for_prediction = None

    with tab_draw:
        try:
            from streamlit_drawable_canvas import st_canvas

            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None and st.button("Predict drawn digit"):
                img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
                image_for_prediction = img
        except ImportError:
            st.info(
                "The drawing canvas requires the `streamlit-drawable-canvas` "
                "package (see requirements.txt). Please use the "
                "**Upload an image** tab instead, or install the package "
                "and restart the app."
            )

    with tab_upload:
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image_for_prediction = Image.open(uploaded_file)
            st.image(image_for_prediction, caption="Uploaded image", width=150)
            if st.button("Predict uploaded digit"):
                pass  # image_for_prediction is already set; button just confirms intent
            else:
                image_for_prediction = image_for_prediction  # keep for auto predict below

    if image_for_prediction is not None:
        model = load_trained_model()
        input_tensor, display_arr = preprocess(image_for_prediction)
        probs = model.predict(input_tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        col1, col2 = st.columns(2)
        with col1:
            st.image(display_arr, caption="Model input (28x28)", width=140, clamp=True)
        with col2:
            st.metric("Predicted Digit", predicted_digit)
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.subheader("Class probabilities")
        st.bar_chart({"probability": {str(i): float(p) for i, p in enumerate(probs)}})

    st.divider()
    st.caption(
        "Kinetrexa Software Pvt. Ltd. — AI & Machine Learning Internship | "
        "Capstone Project: Handwritten Digit Recognition"
    )


if __name__ == "__main__":
    main()
