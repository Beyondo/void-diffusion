import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display
def save(img):
    # Save image to Google Drive and a txt file with the settings used to generate it
    dir = '/content/gdrive/MyDrive/' + colab.save_directory
    if not os.path.exists(dir): os.makedirs(dir)
    imgSavePath = "%s/%d-voidops" % (dir, int(time.mktime(datetime.datetime.now().timetuple())))
    img.save(imgSavePath + ".png")
    display(img)
    imgSavePath = imgSavePath.replace("/content/gdrive/MyDrive/", "")
    print("Saved to " + imgSavePath + ".png")
    if colab.save_settings:
        with open(imgSavePath + "-settings.txt", "w") as f:
            for key, value in colab.settings.items():
                f.write("%s: %s \n" % (key, value))
        print("and " + imgSavePath + ".txt")