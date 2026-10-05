"""Astro image classifier — Gradio app for a free Hugging Face Space (CPU).

The Vercel page in web/ calls it through @gradio/client:

    /classify (image) -> {"label": top class, "confidences": [{label, confidence}, ...]}

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
import json
import os
import re
import tempfile
import zipfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gradio as gr
import keras
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.densenet import preprocess_input

from utils.labels import CLASS_NAMES
from utils.preprocessing import feed, prepare_image

MODEL_REPO = os.getenv("MODEL_REPO", "MA29/astro-image-classifier")
MODEL_FILE = os.getenv("MODEL_FILE", "ensemble_model.keras")


def _load(path):
    custom = {"preprocess_input": preprocess_input}
    try:
        return tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    except (TypeError, ValueError):
        return tf.keras.models.load_model(
            path, custom_objects=custom, compile=False, safe_mode=False
        )


# Keras error messages that name a config key the installed version doesn't know.
_UNKNOWN_KEY = [
    re.compile(r"unexpected keyword argument '(\w+)'"),
    re.compile(r"Unrecognized keyword arguments? passed to \w+: \{'(\w+)'"),
]


def _drop_key(obj, key):
    if isinstance(obj, dict):
        return {k: _drop_key(v, key) for k, v in obj.items() if k != key}
    if isinstance(obj, list):
        return [_drop_key(v, key) for v in obj]
    return obj


def load_model():
    """The model was saved with Keras 3.15, which needs Python 3.11. On an older
    Keras (e.g. a Python 3.10 Space) loading fails on config options added
    since, such as `input_axes` on initializers or `quantization_config` on
    Dense. Those are left at their defaults, so each one the installed Keras
    rejects is removed from config.json and the load retried."""
    keras.mixed_precision.set_global_policy("float32")  # saved as mixed_float16
    path = original = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    dropped = []
    while True:
        try:
            model = _load(path)
            if path != original:
                os.remove(path)  # weights are in memory now
            if dropped:
                print(f"Keras {keras.__version__}: ignored newer config keys {dropped}")
            return model
        except (TypeError, ValueError) as e:
            key = next((m.group(1) for p in _UNKNOWN_KEY for m in [p.search(str(e))] if m), None)
            if key is None or key in dropped or len(dropped) >= 20:
                raise
            dropped.append(key)
            stripped = _strip_config(path, key)
            if path != original:
                os.remove(path)  # the previous stripped copy
            path = stripped


def _strip_config(src, key):
    fd, dst = tempfile.mkstemp(suffix=".keras")
    os.close(fd)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(_drop_key(json.loads(data), key)).encode()
            zout.writestr(item, data)
    return dst


# Loaded once at startup, so the first visitor doesn't wait for the 267 MB download.
model = load_model()


def classify(image):
    if image is None:
        raise gr.Error("Upload an image.")
    preds = model.predict(feed(model, prepare_image(image)), verbose=0)[0]
    return {name.replace("_", " "): float(p) for name, p in zip(CLASS_NAMES, preds)}


with gr.Blocks(title="Astro Image Classifier") as demo:
    gr.Markdown(
        "# Astro Image Classifier\n"
        "VGG19 + DenseNet201 ensemble over 11 astronomical classes. "
        "[Code](https://github.com/Majd1029/Astro-Image-Classifier) · "
        f"[Weights](https://huggingface.co/{MODEL_REPO})"
    )
    with gr.Row():
        image = gr.Image(label="Astronomical image", type="pil")
        label = gr.Label(label="Prediction")
    run = gr.Button("Classify", variant="primary")
    run.click(classify, inputs=image, outputs=label, api_name="classify")

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
