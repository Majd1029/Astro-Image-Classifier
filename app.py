import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from utils.preprocessing import prepare_image
from utils.labels import CLASS_NAMES

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def vgg_preprocess(x):
    return vgg19_preprocess_input(x)

@register_keras_serializable()
def densenet_preprocess(x):
    return densenet_preprocess_input(x)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Astro Image Classifier",
    layout="centered"
)

st.title("🌌 Astro Image Classification")
st.write("Upload an astronomical image and get a prediction.")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/ensemble_model.keras",
        custom_objects={
            "preprocess_input": densenet_preprocess,
        },
        compile=False
    )
model = load_model()

# -------------------------
# Image uploader
# -------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        vgg_img, densenet_img = prepare_image(image)

        preds = model.predict([vgg_img, densenet_img])[0]

        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx]

        predicted_class = CLASS_NAMES[pred_idx]

    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # -------------------------
    # Probability chart
    # -------------------------
    st.subheader("Class probabilities")
    prob_dict = {
        CLASS_NAMES[i]: float(preds[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)