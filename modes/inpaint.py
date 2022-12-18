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
    # Load image
    init_image = Image.open(BytesIO(requests.get(colab.settings['InitialImageURL']).content)).convert('RGB')
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_image = Image.open(BytesIO(requests.get(colab.settings['MaskImageURL']).content)).convert("L")
    mask_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_applied_image = init_image.copy()
    mask_applied_image = Image.blend(mask_applied_image, mask_image, 0.5)
    # rgba
    image_rgba_mask_removed = init_image.convert("RGBA")
    print("Apply mask to image: ", image_rgba_mask_removed.size, mask_image.size)
    image_rgba_mask_removed.putalpha(mask_image)
    display(colab.image_grid([init_image, mask_image, mask_applied_image, image_rgba_mask_removed], 1, 4))
    # Process image
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        latents = None
        if True:
            # generate random image latents for inpainting
            latents = torch.randn(1, 4, 64, 64, device="cuda")
            # blend the mask into the latents
            latents = latents * (1 - mask_image.convert("L").resize((64, 64), Image.BILINEAR).convert("RGB"))
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