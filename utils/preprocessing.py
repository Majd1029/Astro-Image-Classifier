import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

IMG_SIZE = (224, 224)

def prepare_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    vgg_img = vgg_preprocess(img_array.copy())
    densenet_img = densenet_preprocess(img_array.copy())

    return vgg_img, densenet_img
