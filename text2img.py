import torch
import colab
import os
def process(seed, positive_prompt, negative_prompt, guidance_scale, save_to_google_drive, directory):
    genSeed = torch.random.seed() if Seed == 0 else Seed
    generator = torch.Generator("cuda").manual_seed(genSeed)
    image = colab.text2img(Positive, guidance_scale=Guidance_Scale, generator=generator).images[0]
    if SaveToGoogleDrive:
        dir = '/content/gdrive/MyDrive/' + Directory
        if not os.path.exists(dir): os.makedirs(dir)
        imgSavePath = "%s/voidops-%d.png" % (dir, genSeed)
        image.save(imgSavePath)
        print("Saved to " + imgSavePath)