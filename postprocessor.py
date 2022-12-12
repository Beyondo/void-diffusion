import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display

def get_save_path(filename):
    dir = '/content/gdrive/MyDrive/' + colab.save_directory
    if not os.path.exists(dir): os.makedirs(dir)
    return "%s/%s" % (dir, filename)

def save_gdrive(img, filename):
    imgSavePath = get_save_path(filename)
    imgFile = imgSavePath + ".png"
    img.save(imgFile)
    return imgFile.replace("/content/gdrive/MyDrive/", "")

def save_settings(filename, mode):
    imgSavePath = get_save_path(filename)
    settingsFile = imgSavePath + "-settings.txt"
    if colab.save_settings:
        with open(settingsFile, "w") as f:
            if mode == "text2img":
                f.write("Mode: Text to Image\n")
                f.write("Seed: %s\n" % colab.settings['Seed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                f.write("Guidance Scale: %s\n" % colab.settings['GuidanceScale'])
                f.write("Steps: %s\n" % colab.settings['Steps'])
            elif mode == "Image to Image":
                f.write("Mode: Image2Image\n")
                f.write("Seed: %s\n" % colab.settings['Seed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                f.write("Guidance Scale: %s\n" % colab.settings['GuidanceScale'])
                f.write("Strength: %s\n" % colab.settings['Strength'])
                f.write("Initial Image URL: %s\n" % colab.settings['InitialImageURL'])
                f.write("Steps: %s\n" % colab.settings['Steps'])
            f.write(('-' * 64) + "\n")
            f.write("Main Colab: https://colab.research.google.com/drive/1MRwvDBNc4jhlEXSAtdLe49A4C1k35pgp\n")
            f.write("Website: https://voidops.com\n")
    return settingsFile.replace("/content/gdrive/MyDrive/", "")

def post_process(img, filename):
    imgSavePath = get_save_path(filename)
    imgFile = imgSavePath + "-2x.png"