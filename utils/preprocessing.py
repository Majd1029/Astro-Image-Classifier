"""Input preparation for the saved ensemble.

IMPORTANT: the saved model preprocesses internally. The densenet branch begins
with a Lambda(preprocess_input); the vgg branch begins with a GetItem/Stack/Add
chain that performs the VGG channel swap and mean subtraction. Feeding inputs
that have already been through vgg19/densenet `preprocess_input` applies it
twice.

Measured on 110 images (10 per class) with verify_model.py:

    raw 0-255       100.0% correct, mean confidence 99.1%
    pre-processed    71.8% correct, mean confidence 52.3%

Damage is uneven — venus 0%, neptune 10%, mars 20%, while other classes hold at
100%. So: pass raw 0-255 RGB.
"""
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 and return a raw 0-255 float32 batch of shape
    (1, 224, 224, 3). Do not apply preprocess_input to this."""
    image = image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(image, dtype="float32"), axis=0)


def feed(model, arr: np.ndarray):
    """Map the batch onto the model's inputs by name, so the vgg and densenet
    branches cannot be transposed."""
    names = [t.name.split(":")[0].split("/")[0] for t in model.inputs]
    return arr if len(names) == 1 else {n: arr for n in names}
