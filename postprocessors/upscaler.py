import torch, os
import numpy as np
import PIL.Image
import io
import base64
import IPython.display
import IPython
import math
import time
import colab
import postprocessor
import importlib
def upscale(upscaler, scale, image_input_path):
    image = PIL.Image.open(image_input_path)
    # Upscale image
    if upscaler.lower() == "bicubic":
        image = image.resize((image.width * scale, image.height * scale), PIL.Image.BICUBIC)
    elif upscaler.lower() == "gfpgan":
        os.makedirs("temp/input", exist_ok=True)
        os.makedirs("temp/output", exist_ok=True)
        image.save("temp/input/upload.png")
        IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i temp/input -o temp/output -v 1.3.8 -s " + str(scale) + " --bg_upsampler realesrgan")
        image = PIL.Image.open("temp/output.png")
    # Convert back to PIL image
    return image