import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display

def get_save_name(filename):
    dir = '/content/gdrive/MyDrive/' + colab.save_directory
    if not os.path.exists(dir): os.makedirs(dir)
    return "%s/%d" % (dir, filename)

def save_gdrive(img, filename):
    imgSavePath = get_save_name()
    imgFile = imgSavePath + ".png"
    img.save(imgFile)
    return imgFile.replace("/content/gdrive/MyDrive/", "")

def save_settings(filename):
    imgSavePath = get_save_name()
    settingsFile = imgSavePath + "-settings.txt"
    if colab.save_settings:
        with open(settingsFile, "w") as f:
            for key, value in colab.settings.items():
                f.write("%s: %s \n" % (key, value))
    return settingsFile.replace("/content/gdrive/MyDrive/", "")

def post_process(img, filename):
    imgSavePath = get_save_name()
    imgFile = imgSavePath + "-2x.png"