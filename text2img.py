import torch, os, datetime, colab
from IPython.display import Image
from IPython.display import display
def process(seed, positive_prompt, negative_prompt, guidance_scale, inference_steps, save_to_google_drive, directory):
    genSeed = torch.random.seed() if seed == 0 else seed
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.text2img(positive_prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=inference_steps, generator=generator).images[0]
    if save_to_google_drive:
        dir = '/content/gdrive/MyDrive/' + directory
        if not os.path.exists(dir): os.makedirs(dir)
        num = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
        imgSavePath = "%s/%d-voidops" % (dir, num)
        image.save(imgSavePath + ".png")
        print("Saved to " + imgSavePath)
        display(image)
        return image