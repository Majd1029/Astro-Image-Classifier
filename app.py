"""Astro image classifier — Streamlit app for Streamlit Community Cloud.

Weights are pulled from the Hugging Face model repo at startup
(MA29/astro-image-classifier), so this GitHub repo stays small and nothing
large is committed.

Architecture, read directly from the saved ensemble_model.keras:

    vgg_ensemble_input      [None,224,224,3] -> vgg_branch_model      -> softmax(11)
    densenet_ensemble_input [None,224,224,3] -> densenet_branch_model -> softmax(11)
                                             -> Average

Two things the saved graph dictates:

1. `custom_objects={"preprocess_input": ...}` is REQUIRED. The densenet branch
   holds a Lambda serialised as {"config": "preprocess_input"}, so Keras
   resolves that function by name from custom_objects at load time.

2. Both branches preprocess INTERNALLY — densenet via that Lambda, vgg via a
   GetItem/Stack/Add chain (channel swap + mean subtract). Both inputs therefore
   take RAW 0-255 RGB. Preprocessing before feeding does it twice: measured at
   71.8% correct versus 100% on the same images. See verify_model.py.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Astro Image Classifier", page_icon="🌌", layout="centered")

MODEL_REPO = os.getenv("MODEL_REPO", "MA29/astro-image-classifier")
MODEL_FILE = os.getenv("MODEL_FILE", "ensemble_model.keras")
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "black_hole", "earth", "galaxy", "jupiter", "mars", "mercury",
    "neptune", "pluto", "saturn", "uranus", "venus",
]


@st.cache_resource(show_spinner=False)
def get_model():
    """Loaded once per container. TensorFlow is imported lazily so the page
    renders before the ~600MB import and 267MB download begin."""
    import keras
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.applications.densenet import preprocess_input

    keras.mixed_precision.set_global_policy("float32")  # saved as mixed_float16

    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    custom = {"preprocess_input": preprocess_input}
    try:
        model = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    except (TypeError, ValueError):
        model = tf.keras.models.load_model(
            path, custom_objects=custom, compile=False, safe_mode=False
        )
    names = [t.name.split(":")[0].split("/")[0] for t in model.inputs]
    return model, names


def classify(image: Image.Image):
    model, names = get_model()
    arr = np.expand_dims(
        np.array(image.convert("RGB").resize(IMG_SIZE), dtype="float32"), 0
    )  # raw 0-255 — the graph preprocesses internally
    feed = arr if len(names) == 1 else {n: arr for n in names}
    return model.predict(feed, verbose=0)[0]


st.title("🌌 Astro Image Classifier")
st.caption(
    "VGG19 + DenseNet201 ensemble over 11 astronomical classes. "
    "[Code](https://github.com/Majd1029/Astro-Image-Classifier) · "
    f"[Weights](https://huggingface.co/{MODEL_REPO})"
)

examples_dir = "examples"
example_files = (
    sorted(f for f in os.listdir(examples_dir) if f.lower().endswith((".jpg", ".png")))
    if os.path.isdir(examples_dir)
    else []
)

image = None
tab_upload, tab_example = st.tabs(["Upload an image", "Try an example"])

with tab_upload:
    up = st.file_uploader("Astronomical image", type=["jpg", "jpeg", "png"])
    if up:
        image = Image.open(up)

with tab_example:
    if example_files:
        choice = st.selectbox(
            "Example", example_files, format_func=lambda f: os.path.splitext(f)[0]
        )
        cols = st.columns(len(example_files))
        for col, f in zip(cols, example_files):
            col.image(os.path.join(examples_dir, f), caption=os.path.splitext(f)[0])
        if st.button("Classify this example"):
            image = Image.open(os.path.join(examples_dir, choice))
    else:
        st.info("No examples bundled with this deployment.")

if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    with st.spinner("Loading model and classifying (first run downloads 267MB)..."):
        preds = classify(image)

    order = np.argsort(preds)[::-1]
    top = order[0]

    st.success(f"**{CLASS_NAMES[top].replace('_', ' ')}** — {preds[top]:.1%} confidence")

    st.subheader("Class probabilities")
    st.bar_chart(
        {CLASS_NAMES[i]: float(preds[i]) for i in order[:5]},
        horizontal=True,
        height=220,
    )

    with st.expander("All 11 classes"):
        for i in order:
            st.write(f"{CLASS_NAMES[i]:<12} {preds[i]:.2%}")

st.divider()
st.caption(
    "Closed-world over 11 classes — every input is forced into one of them, so "
    "out-of-distribution images get confident, meaningless labels."
)
