import torch, os, time, datetime, colab, postprocessor, importlib
from IPython.display import Image
from IPython.display import display
importlib.reload(postprocessor)
def process(ShouldSave):
    # Load image
    image = Image.open(colab.settings['ImageURL'])
    # Process image
    genSeed = torch.random.seed() if seed == 0 else seed
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.img2img(
        width=width,
        height=height,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        generator=generator).images[0]
    if ShouldSave:
        postprocessor.save(image)