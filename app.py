"""
app.py
========================================================
Streamlit Web Application — CNN Digit Classifier
========================================================

Deploys the trained CNN from Notebook 1A as an interactive web app.
Users can either:
  1) Draw a digit (0-9) directly on an on-screen canvas, or
  2) Upload an image file of a handwritten digit
and get a real-time prediction with class-probability confidence scores.

To run locally:
    streamlit run app.py

Requirements (see requirements.txt):
    streamlit
    streamlit-drawable-canvas
    tensorflow
    numpy
    pillow
    pandas

The app expects a trained Keras model file named 'digit_cnn_model.keras'
in the same directory (produced by train_model.py / Notebook 1A).
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CNN Digit Classifier",
    page_icon="🔢",
    layout="centered",
)


# ----------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------
MODEL_PATH = "digit_cnn_model.keras"


@st.cache_resource
def load_model(path: str):
    """Load the trained Keras CNN model. Cached across reruns."""
    try:
        model = keras.models.load_model(path)
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


model, load_error = load_model(MODEL_PATH)


# ----------------------------------------------------------------------
# Image preprocessing helpers
# ----------------------------------------------------------------------
def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """
    Convert an arbitrary PIL image into the 8x8 grayscale, normalized,
    channel-expanded array the model expects: shape (1, 8, 8, 1).
    """
    # Convert to grayscale
    img = img.convert("L")

    # Resize to 8x8 (the resolution the model was trained on)
    img = img.resize((8, 8), Image.LANCZOS)

    arr = np.array(img).astype("float32")

    # Heuristic: if background is bright (white paper) and digit is dark,
    # invert so the digit is bright-on-dark like the training data.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Scale to [0, 16] range to match training normalization (digits data
    # was originally 0-16), then normalize to [0, 1] the same way the
    # model was trained.
    arr = (arr / 255.0) * 16.0
    arr = arr / 16.0

    arr = np.expand_dims(arr, axis=(0, -1))  # (1, 8, 8, 1)
    return arr


def predict(arr: np.ndarray):
    """Run the model on a preprocessed (1, 8, 8, 1) array."""
    probs = model.predict(arr, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    return pred_class, probs


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app deploys a Convolutional Neural Network (CNN) trained "
        "in Notebook 1A to classify handwritten digits (0-9)."
    )
    st.write(
        "The model was trained on 8x8 grayscale digit images "
        "(scikit-learn's `digits` dataset)."
    )
    if model is not None:
        st.success("Model loaded ✅")
    else:
        st.error("Model failed to load ❌")

    st.divider()
    input_mode = st.radio(
        "Choose input method",
        ["Draw a digit", "Upload an image"] if CANVAS_AVAILABLE else ["Upload an image"],
    )

    st.divider()
    st.caption(
        "Note: because the underlying model was trained on low-resolution "
        "8x8 images, predictions work best on simple, centered, bold "
        "digit strokes."
    )


# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("🔢 CNN Digit Classifier")
st.write(
    "Draw a digit or upload an image below, and the CNN will predict "
    "which digit (0-9) it is, along with a confidence score for every class."
)

if load_error:
    st.error(f"Could not load the model file '{MODEL_PATH}'.\n\nError: {load_error}")
    st.info("Run `python train_model.py` to generate the model file, then restart the app.")
    st.stop()

input_array = None
display_image = None

if input_mode == "Draw a digit" and CANVAS_AVAILABLE:
    st.subheader("Draw a digit (0–9)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
        rgba = canvas_result.image_data.astype("uint8")
        display_image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        input_array = preprocess_pil_image(display_image)

else:
    st.subheader("Upload a digit image")
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPG image of a single handwritten digit",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        display_image = Image.open(uploaded_file)
        input_array = preprocess_pil_image(display_image)


# ----------------------------------------------------------------------
# Prediction & results
# ----------------------------------------------------------------------
if input_array is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Input preview**")
        st.image(display_image, width=200)
        st.caption("Model sees a downsampled 8x8 grayscale version of this image.")

    pred_class, probs = predict(input_array)
    confidence = probs[pred_class] * 100

    with col2:
        st.write("**Prediction**")
        st.metric(label="Predicted Digit", value=str(pred_class))
        st.metric(label="Confidence", value=f"{confidence:.1f}%")

    st.write("**Class probabilities**")
    prob_df = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability": probs,
    }).set_index("Digit")
    st.bar_chart(prob_df)

else:
    st.info("Draw or upload a digit above to see a prediction.")

st.divider()
st.caption(
    "Built with Streamlit + TensorFlow/Keras. Model architecture, training, "
    "and evaluation details are in Notebook 1A (CNN Algorithm Notebook)."
)
