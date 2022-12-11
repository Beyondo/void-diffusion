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

def save_settings(filename):
    imgSavePath = get_save_path(filename)
    settingsFile = imgSavePath + "-settings.txt"
    if colab.save_settings:
        with open(settingsFile, "w") as f:
            for key, value in colab.settings.items():
                f.write("%s: %s\n" % (key, value))
            f.write(('-' * 64)  + "\nMain Colab: https://colab.research.google.com/drive/1MRwvDBNc4jhlEXSAtdLe49A4C1k35pgp\n")
            f.write("Website: https://voidops.com\n")
    return settingsFile.replace("/content/gdrive/MyDrive/", "")

def post_process(img, filename):
    imgSavePath = get_save_path(filename)
    imgFile = imgSavePath + "-2x.png"