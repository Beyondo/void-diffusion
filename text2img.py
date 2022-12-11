import torch, os, time, datetime, colab, postprocessor, importlib
from IPython.display import Image
from IPython.display import display
importlib.reload(postprocessor)
def process(ShouldSave):
    if 'Seed' not in colab.settings: "Please set your settings first."
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
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave:
        if colab.save_settings: postprocessor.save_settings(timestamp)
        path = postprocessor.save_gdrive(image, timestamp)
        display(image)
        print("Saved to " + path)
    postprocessor.post_process(image, timestamp)