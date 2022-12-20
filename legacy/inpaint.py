import torch, os, time, datetime, importlib
from legacy import colab, postprocessor, progress
from IPython.display import Image
from IPython.display import display

import requests
from PIL import Image
from io import BytesIO
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, ShouldPreview = True, ReplaceResult = True):
    progress.replace_result = ReplaceResult
    colab.prepare("inpaint")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if colab.save_settings: postprocessor.save_settings(timestamp, mode="inpaint")
    # Load image
    init_image = None
    if colab.settings['UseLastOutputAsInitialImage'] and colab.last_generated_image is not None:
        init_image = colab.last_generated_image
    else:
        init_image = Image.open(BytesIO(requests.get(colab.settings['InitialImageURL']).content)).convert('RGB')
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_image = Image.open(BytesIO(requests.get(colab.settings['MaskImageURL']).content)).convert("RGB")
    init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
    mask_applied_image = Image.blend(init_image, mask_image, 0.5)
    display(colab.image_grid([init_image, mask_image, mask_applied_image], 1, 3))
    colab.image_size = init_image.size
    init_image = init_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    grey_mask = mask_image.convert("L")
    # Process image
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        latents = None
        if False:
            # generate random image latents for inpainting
            latents = torch.randn(1, 4, 64, 64, device="cuda")
            # blend the mask into the latents
            latents = latents * (1 - mask_image.convert("L").resize((64, 64), Image.BILINEAR).convert("RGB"))
        image = colab.inpaint(
            prompt=colab.settings['Prompt'],
            image=init_image,
            mask_image=grey_mask,
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
            num_inference_steps=colab.settings['Steps'],
            generator=generator,
            callback=progress.callback if ShouldPreview else None,
            callback_steps=20).images[0]
        # convert the image back to the original size
        image = image.resize(colab.image_size)
        colab.last_generated_image = image
        progress.show(image)
        postprocessor.post_process(image, "%d_%d" % (timestamp, i), colab.get_current_image_uid(), ShouldSave, ReplaceResult)
        display("Iterations: %d/%d" % (i + 1,  num_iterations), display_id="iterations")
    postprocessor.join()