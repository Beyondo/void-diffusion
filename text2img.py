import torch, os, time, datetime, colab, postprocessor
from IPython.display import Image
from IPython.display import display
def process(ShouldSave):
    genSeed = torch.random.seed() if colab.settings['Seed'] == 0 else colab.settings['Seed']
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.text2img(
        width=colab.settings['Width'],
        height=colab.settings['Height'],
        prompt=colab.settings['Prompt'],
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        num_inference_steps=colab.settings['Steps'],
        generator=generator).images[0]
    if save:
        postprocessor.save_image(image)
    