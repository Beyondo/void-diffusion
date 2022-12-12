import torch, os, time, datetime, colab, postprocessor, progress, importlib, random
from IPython.display import display
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
    # how to get the image every 100 steps in StableDiffusionPipeline?
    images = colab.text2img(
        width=colab.settings['Width'],
        height=colab.settings['Height'],
        prompt=colab.settings['Prompt'],
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        num_inference_steps=colab.settings['Steps'],
        num_images_per_prompt=colab.settings['NumImages'],
        generator=generator,
        callback=progress.callback,
        callback_steps=10).images
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave:
        if colab.save_settings: postprocessor.save_settings(timestamp, mode="text2img")
        for i, image in enumerate(images):
            imageName = "%d_%d" % (timestamp, i)
            path = postprocessor.save_gdrive(image, imageName)
            display(image, display_id=str(i))
            print("Saved to " + path)
            postprocessor.post_process(image, imageName)