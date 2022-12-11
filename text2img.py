import torch, os, time, datetime, colab, postprocessor, importlib
from IPython.display import Image
from IPython.display import display
importlib.reload(postprocessor)
def processing_callback(iter, t, latents):
    # convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = colab.text2img.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # convert to PIL Images
        image = colab.text2img.numpy_to_pil(image)
        # do something with the Images
        for i, img in enumerate(image):
            display(img)

def process(ShouldSave):
    if 'Seed' not in colab.settings: "Please set your settings first."
    genSeed = torch.random.seed() if colab.settings['Seed'] == 0 else colab.settings['Seed']
    generator = torch.Generator("cuda").manual_seed(genSeed)
    # how to get the image every 100 steps in StableDiffusionPipeline?
    image = colab.text2img(
        width=colab.settings['Width'],
        height=colab.settings['Height'],
        prompt=colab.settings['Prompt'],
        negative_prompt=colab.settings['NegativePrompt'],
        guidance_scale=colab.settings['GuidanceScale'],
        num_inference_steps=colab.settings['Steps'],
        generator=generator,
        callback=processing_callback,
        callback_steps=10).images[0]
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave:
        if colab.save_settings: postprocessor.save_settings(timestamp)
        path = postprocessor.save_gdrive(image, timestamp)
        display(image)
        print("Saved to " + path)
    postprocessor.post_process(image, timestamp)