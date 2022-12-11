import torch, os, time, datetime, colab, postprocessor, progress, importlib
from IPython.display import Image
from IPython.display import display

import requests
from PIL import Image
from io import BytesIO
importlib.reload(postprocessor)
def process(ShouldSave):
    # Load image
    response = requests.get(colab.settings['ImageURL'])
    init_image = Image.open(BytesIO(response.content)).convert('RGB')
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    display(init_image)
    # Process image
    colab.settings['Seed'] = torch.random.seed() if colab.settings['Seed'] == 0 else colab.settings['Seed']
    generator = torch.Generator("cuda").manual_seed(colab.settings['Seed'])
    image = colab.img2img(
        prompt=colab.settings['Prompt'],
        image=init_image,
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        strength=colab.settings['Strength'],
        num_inference_steps=colab.settings['Steps'],
        generator=generator,
        callback=progress.callback,
        callback_steps=10).images[0]
    if ShouldSave:
        postprocessor.save_gdrive(image)