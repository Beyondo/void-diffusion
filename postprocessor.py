import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display
def save_image(img, diffusion_settings):
    # Save image to Google Drive and a txt file with the settings used to generate it
    dir = '/content/gdrive/MyDrive/' + diffusion_settings['directory']
    if not os.path.exists(dir): os.makedirs(dir)
    imgSavePath = "%s/%d-voidops" % (dir, int(time.mktime(datetime.datetime.now().timetuple())))
    img.save(imgSavePath + ".png")
    with open(imgSavePath + ".txt", "w") as f:
        for key, value in diffusion_settings.items():
            f.write("%s: %s \n" % (key, value))
    display(img)
    print("Saved to " + imgSavePath + ".png")
    return img