import torch, os, time, datetime, colab
from IPython.display import Image
from IPython.display import display
def save_gdrive(img):
    # Save image to Google Drive and a txt file with the settings used to generate it
    dir = '/content/gdrive/MyDrive/' + colab.save_directory
    if not os.path.exists(dir): os.makedirs(dir)
    imgSavePath = "%s/%d-voidops" % (dir, int(time.mktime(datetime.datetime.now().timetuple())))
    imgFile = imgSavePath + ".png"
    settingsFile = imgSavePath + "-settings.txt"
    img.save(imgFile) # Save before displaying, so that the image is saved regardless of whether the user sees it
    # Save settings
    if colab.save_settings:
        with open(settingsFile, "w") as f:
            for key, value in colab.settings.items():
                f.write("%s: %s \n" % (key, value))
    # Display image
    display(img)
    # Print save locations
    print("Saved to " + imgFile.replace("/content/gdrive/MyDrive/", ""))
    if colab.save_settings: print("and " + settingsFile.replace("/content/gdrive/MyDrive/", ""))