import torch
import colab
import os
from IPython.display import Image
def process(seed, positive_prompt, negative_prompt, guidance_scale, inference_steps, save_to_google_drive, directory):
    genSeed = torch.random.seed() if seed == 0 else seed
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.text2img(positive_prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=inference_steps, generator=generator).images[0]
    if save_to_google_drive:
        dir = '/content/gdrive/MyDrive/' + directory
        if not os.path.exists(dir): os.makedirs(dir)
        imgSavePath = "%s/voidops-%d.png" % (dir, genSeed)
        image.save(imgSavePath)
        print("Saved to " + imgSavePath)
        return image