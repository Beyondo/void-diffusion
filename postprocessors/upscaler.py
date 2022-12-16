import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
import io
import base64
import IPython.display
import math
import time
import colab
import postprocessor
import importlib
def upscale(upscaler, scale, image_input_path):
    image = TF.to_tensor(PIL.Image.open(image_input_path))
    # Upscale image
    if upscaler.lower() == "bicubic":
        image = TF.to_pil_image(image)
        image = image.resize((image.width * scale, image.height * scale), PIL.Image.BICUBIC)
        image = TF.to_tensor(image)
        image = image.unsqueeze(0)
    # Convert back to PIL image
    image = TF.to_pil_image(image[0])
    return image