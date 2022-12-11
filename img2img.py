import torch, os, time, datetime, colab, postprocessor, importlib
from IPython.display import Image
from IPython.display import display
importlib.reload(postprocessor)
def process(ShouldSave):
    # Load image
    image = Image.open(colab.settings['ImageURL'])
    display(image) # temporary
    # Process image
    genSeed = torch.random.seed() if colab.settings['Seed'] == 0 else colab.settings['Seed']
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.img2img(
        width=colab.settings['Width'],
        height=colab.settings['Height'],
        prompt=colab.settings['Prompt'],
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        num_inference_steps=colab.settings['Steps'],
        generator=generator).images[0]
    if ShouldSave:
        postprocessor.save_gdrive(image)