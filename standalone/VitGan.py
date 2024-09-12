
import tensorflow as tf
import numpy as np
from PIL import Image
from .models import Generator  # local import
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
ff_dim = 32
num_heads = 2
patch_size = 8
projection_dim = 64
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
num_patches = (IMG_HEIGHT // patch_size) ** 2


# Define a function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


class VitGan():
    def __init__(self, config):
        # Load the model
        self.model = Generator(input_shape, patch_size, num_patches, projection_dim, num_heads, ff_dim)
        self.model.load_weights(config["model_url"])

    def infer(self, img_path):
        # Preprocess the input image
        input_image = preprocess_image(img_path)

        # Perform inference
        predictions = self.model.predict(input_image)

        return predictions

