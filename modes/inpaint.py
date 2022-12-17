import torch, os, time, datetime, colab, postprocessor, progress, importlib
from IPython.display import Image
from IPython.display import display

import requests
from PIL import Image
from io import BytesIO
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, ShouldPreview = True):
    colab.prepare("inpaint")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if colab.save_settings: postprocessor.save_settings(timestamp, mode="inpaint")
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    # Load image
    init_image = Image.open(BytesIO(requests.get(colab.settings['InitialImageURL']).content)).convert('RGB')
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_image = Image.open(BytesIO(requests.get(colab.settings['MaskImageURL']).content)).convert('RGBA')
    mask_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_applied_image = Image.new("RGB", init_image.size)
    mask_applied_image.paste(init_image, (0, 0), mask_image)
    latents = None
    if True: #colab.settings["Strength"] > 0:
        # generate latent space from the mask using torch
        latent = torch.randn(1, 512, device="cuda")
        # generate a mask from the mask image
        mask = torch.tensor(mask_image, device="cuda").float() / 255
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        # apply the mask to the latent space
        latent_image = latent * mask + (1 - mask) * 0.5
        latent_image = latent_image.permute(0, 2, 3, 1).squeeze(0)
        latent_image = latent_image.cpu().numpy()
        latent_image = Image.fromarray((latent_image * 255).astype("uint8"))
        latent_image = latent_image.resize(init_image.size)
        latents = [latent]
        grid = colab.image_grid([init_image, mask_image, mask_applied_image, latent_image], 1, 4)
    else:
        grid = colab.image_grid([init_image, mask_image, mask_applied_image, latent_image], 1, 3)
    display(grid)
    # Process image
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        image = colab.inpaint(
            prompt=colab.settings['Prompt'],
            image=init_image,
            mask_image=mask_image,
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
            num_inference_steps=colab.settings['Steps'],
            generator=generator,
            latents=latents,
            callback=progress.callback if ShouldPreview else None,
            callback_steps=20).images[0]
        progress.show(image)
        postprocessor.post_process(image, "%d_%d" % (timestamp, i), ShouldSave)
        display("Iterations: %d/%d" % (i + 1,  num_iterations), display_id="iterations")