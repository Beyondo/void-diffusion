import torch, os, time, datetime, colab, postprocessor
from IPython.display import Image
from IPython.display import display
def process(width, height, seed, positive_prompt, negative_prompt, guidance_scale, inference_steps, save_to_google_drive, directory):
    global Seed
    print ("Generating image using seed: " + str(Seed) + "...")
    genSeed = torch.random.seed() if seed == 0 else seed
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.text2img(
        width=width,
        height=height,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        generator=generator).images[0]
    if save_to_google_drive:
        postprocessor.save_image(image)
    