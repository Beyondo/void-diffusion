import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display
def process(width, height, seed, positive_prompt, negative_prompt, strength, image_url, guidance_scale, inference_steps, save_to_google_drive, directory):
    # Load image
    image = Image.open(image_url)
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
    if save_to_google_drive:
        dir = '/content/gdrive/MyDrive/' + directory
        if not os.path.exists(dir): os.makedirs(dir)
        imgSavePath = "%s/%d-voidops" % (dir, int(time.mktime(datetime.datetime.now().timetuple())))
        image.save(imgSavePath + ".png")
        display(image)
        print("Saved to " + imgSavePath + ".png")
        return image
