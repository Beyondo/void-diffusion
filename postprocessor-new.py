import torch, os, time, importlib
from legacy import colab
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML
import PIL, base64
import upscaler
importlib.reload(upscaler)
def get_save_path(filename):
    dir = '/content/gdrive/MyDrive/' + colab.save_directory
    if not os.path.exists(dir): os.makedirs(dir)
    return "%s/%s" % (dir, filename)

def save_gdrive(img, filename):
    imgSavePath = get_save_path(filename)
    imgFile = imgSavePath + ".png"
    img.save(imgFile)
    return imgFile.replace("/content/gdrive/MyDrive/", "")

def write_general_settings(f):
    f.write("Guidance Scale: %s\n" % colab.settings['GuidanceScale'])
    f.write("Steps: %s\n" % colab.settings['Steps'])
    f.write("Iterations: %s\n" % colab.settings['Iterations'])
    f.write("Generated seeds: %d (0)" % colab.settings['InitialSeed'])
    for i in range(1, colab.settings['Iterations']):
        f.write(", %d (%d)" % (colab.settings['InitialSeed'] + i, i))
    f.write("\n")
    
def save_settings(filename, mode):
    imgSavePath = get_save_path(filename)
    settingsFile = imgSavePath + "-settings.txt"
    if colab.save_settings:
        with open(settingsFile, "w") as f:
            if mode == "text2img":
                f.write("Model: %s\n" % colab.model_name)
                f.write("Mode: Text to Image\n")
                f.write("Initial Seed: %s\n" % colab.settings['InitialSeed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                write_general_settings(f)
            elif mode == "img2img":
                f.write("Model: %s\n" % colab.model_name)
                f.write("Mode: Image to Image\n")
                f.write("Initial Seed: %s\n" % colab.settings['InitialSeed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                f.write("Strength: %s\n" % colab.settings['Strength'])
                f.write("Initial Image URL: %s\n" % colab.settings['InitialImageURL'])
                write_general_settings(f)
            elif mode == "inpaint":
                f.write("Model: %s\n" % colab.inpaint_model_name)
                f.write("Mode: Inpainting\n")
                f.write("Initial Seed: %s\n" % colab.settings['InitialSeed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                f.write("Initial Image URL: %s\n" % colab.settings['InitialImageURL'])
                f.write("Mask Image URL: %s\n" % colab.settings['MaskImageURL'])
                write_general_settings(f)
            f.write(('-' * 64) + "\n")
            f.write("Latest version: https://voidops.com/diffusion\n")
    return settingsFile.replace("/content/gdrive/MyDrive/", "")

def display_post_image(img, imageName, image_uid, gdrive, replaceResult):
    display(HTML("<label>Scaled: Processing..."), display_id=image_uid + "_scaled")
    imgSavePath = get_save_path(imageName)
    if colab.settings['Scale'] != "1x":
        scale = int(colab.settings['Scale'][:-1])
        scaled_image = upscaler.upscale(colab.settings['Upscaler'], scale, img)
        if gdrive:
            path = save_gdrive(scaled_image, imageName + "-%dx" % scale)
            # if '/content/gdrive/MyDrive/path exists, then the image was saved to gdrive
            if os.path.exists("/content/gdrive/MyDrive/" + path):
                display(HTML("<label>Saved: %s" % path), display_id=image_uid + "_scaled_saved")
        else:
            scaled_image.save(imgSavePath + "-%dx.png" % scale)
            
        scaled_image.save("media-dir/%s-%dx.png" % (image_uid, scale))
        html_link = "<a href='%s%s-%dx.png' target='_blank'>Full %dx-scaled Image</a>" % (colab.server_url, image_uid, scale, scale)
        display(HTML("<label>Scaled: %s" % html_link), display_id=image_uid + "_scaled")
        if replaceResult:
            scaled_image.thumbnail(img.size, PIL.Image.ANTIALIAS)
            display(scaled_image, display_id=image_uid)
        else:
            display(scaled_image, display_id=image_uid + "_image_scaled")

def process_all(gdrive = True, replaceResult = True):
    ## copy all images from media-dir to temp/input
    #shutil.rmtree("temp/input", ignore_errors=True)
    #shutil.copytree("media-dir", "temp/input")
    ## process all images in temp/input
    #if colab.settings['Scale'] != "1x":
    #    scale = int(colab.settings['Scale'][:-1])
    #    upscaler.upscale_all(colab.settings['Upscaler'], scale)
    ## copy all images from temp/output to media-dir store their names
    #shutil.copytree("temp/output", "media-dir")
    ## loop through all images in the temp/output directory
    #post_images = os.listdir("temp/output")
    #for image in post_images:
    #    # get the image name and uid
    #    image_uid = image.split("-")[0]
    #    imageName = image.split("-")[1]
    #    # get the image path
    #    imgPath = "temp/output/%s" % image
    #    # load the image
    #    img = PIL.Image.open(imgPath)
    #    # process the image
    #    display_post_image(img, imageName, image_uid, gdrive, replaceResult)
    #    # remove the image from the temp/output directory
    #    os.remove(imgPath)


#post_process_jobs = []
#def start_post_processing(img, imageName, image_uid, gdrive, replaceResult):
#    display(HTML("<label>Scaled: Processing..."), display_id=image_uid + "_scaled")
#    imgSavePath = get_save_path(imageName)
#    if colab.settings['Scale'] != "1x":
#        scale = int(colab.settings['Scale'][:-1])
#        scaled_image = upscaler.upscale(colab.settings['Upscaler'], scale, img)
#        if gdrive:
#            path = save_gdrive(scaled_image, imageName + "-%dx" % scale)
#            # if '/content/gdrive/MyDrive/path exists, then the image was saved to gdrive
#            if os.path.exists("/content/gdrive/MyDrive/" + path):
#                display(HTML("<label>Saved: %s" % path), display_id=image_uid + "_scaled_saved")
#        else:
#            scaled_image.save(imgSavePath + "-%dx.png" % scale)
#            
#        scaled_image.save("media-dir/%s-%dx.png" % (image_uid, scale))
#        html_link = "<a href='%s%s-%dx.png' target='_blank'>Full %dx-scaled Image</a>" % (colab.server_url, image_uid, scale, scale)
#        display(HTML("<label>Scaled: %s" % html_link), display_id=image_uid + "_scaled")
#        if replaceResult:
#            scaled_image.thumbnail(img.size, PIL.Image.ANTIALIAS)
#            display(scaled_image, display_id=image_uid)
#        else:
#            display(scaled_image, display_id=image_uid + "_image_scaled")

image_uids = []
def add(image_uid):
    image_uids.append(image_uid)
is_job_queue_running = False
def save(img, imageName, image_uid, gdrive = True):
    global is_job_queue_running, post_process_jobs
    if not os.path.exists("media-dir"):
        os.makedirs("media-dir")
    if gdrive:
        path = save_gdrive(img, imageName)
        display(HTML("<label>Saved: %s" % path), display_id=image_uid + "_saved")
    img.save("media-dir/%s.png" % image_uid) 
    html_link = "<a href='%s%s.png' target='_blank'>Original Image</a>" % (colab.server_url, image_uid)
    display(HTML("<label>Original: %s" % html_link), display_id=image_uid + "_original")