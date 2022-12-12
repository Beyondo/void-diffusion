import torch, os, time, datetime, colab, postprocessor, progress, importlib, random
from IPython.display import Image
from IPython.display import display

import requests
from PIL import Image
from io import BytesIO
importlib.reload(postprocessor)
def process(ShouldSave):
    if 'Seed' not in colab.settings:
        print("Please set your settings first.")
        return
    random.seed(int(time.time_ns()))
    if colab.settings['Seed'] == 0:
        colab.settings['InitialSeed'] = random.getrandbits(64)
    else:
        colab.settings['InitialSeed'] = colab.settings['Seed']
    generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'])
    # Load image
    response = requests.get(colab.settings['InitialImageURL'])
    init_image = Image.open(BytesIO(response.content)).convert('RGB')
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    display(init_image)
    # Process image
    images = colab.img2img(
        prompt=colab.settings['Prompt'],
        image=init_image,
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        strength=colab.settings['Strength'],
        num_inference_steps=colab.settings['Steps'],
        generator=generator,
        callback=progress.callback,
        callback_steps=10).images
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave:
        if colab.save_settings: postprocessor.save_settings(timestamp, mode="img2img")
        for i, image in enumerate(images):
            imageName = "%d_%d" % (timestamp, i)
            path = postprocessor.save_gdrive(image, imageName)
            display(image, display_id=str(i))
            print("Saved to " + path)
            postprocessor.post_process(image, imageName)