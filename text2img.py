import torch, os, time, datetime, colab, postprocessor, progress, importlib

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers.schedulers import PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPTextModel


from IPython.display import display
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, ShouldPreview = True):
    colab.prepare("text2img")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave and colab.save_settings: postprocessor.save_settings(timestamp, mode="text2img")
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        # Tokenize prompt
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        prompt_tokens = tokenizer(colab.settings['Prompt'], return_tensors="pt").input_ids.cuda()
        # Load model
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        # Generate image from prompt
        image = colab.text2img(
            prompt=prompt_tokens,
            model=model,
            width=colab.settings['Width'],
            height=colab.settings['Height'],
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
            num_inference_steps=colab.settings['Steps'],
            generator=generator,
            callback=progress.callback if ShouldPreview else None,
            callback_steps=20).images[0]
        progress.show(image)
        imageName = "%d_%d" % (timestamp, i)
        if ShouldSave:
            path = postprocessor.save_gdrive(image, imageName)
            print("Saved to " + path)
            postprocessor.post_process(image, imageName)
        display("Iterations: %d/%d" % (i + 1,  num_iterations), display_id="iterations")