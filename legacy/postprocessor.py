import torch, os, time, importlib
from legacy import colab
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML
import PIL, base64
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


post_process_jobs = []
def start_post_processing(img, imageName, image_uid, gdrive, replacePreview):
    display(HTML("<label>Scaled: Processing..."), display_id=image_uid + "_scaled")
    imgSavePath = get_save_path(imageName)
    if colab.settings['Scale'] != "1x":
        import upscaler
        importlib.reload(upscaler)
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
        scaled_image.thumbnail(img.size, PIL.Image.ANTIALIAS)
        if replacePreview:
            display(scaled_image, display_id=image_uid)
        else:
            display(scaled_image, display_id=image_uid + ("-%dx" % scale))
    post_process_jobs.pop(0)
import threading

def job_queue():
    while True:
        if len(post_process_jobs) > 0:
            start_post_processing(*post_process_jobs[0])
        else:
            time.sleep(0.1)
def post_process(img, imageName, image_uid, gdrive = True, replacePreview = True):
    if not os.path.exists("media-dir"):
        os.makedirs("media-dir")
    if gdrive:
        path = save_gdrive(img, imageName)
        display(HTML("<label>Saved: %s" % path), display_id=image_uid + "_saved")
    img.save("media-dir/%s.png" % image_uid) 
    html_link = "<a href='%s%s.png' target='_blank'>Original Image</a>" % (colab.server_url, image_uid)
    display(HTML("<label>Original: %s" % html_link), display_id=image_uid + "_original")
    post_process_jobs.append((img, imageName, image_uid, gdrive, replacePreview))

th = threading.Thread(target=job_queue)
def run():
    th.start()

def join():
    while len(post_process_jobs) > 0:
        time.sleep(0.1)
