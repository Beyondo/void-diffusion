import torch, os, time, datetime, importlib
from legacy import colab, postprocessor, progress
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
            f.write("Model: %s\n" % colab.settings['ModelName'])
            if mode == "text2img":
                f.write("Mode: Text to Image\n")
                f.write("Initial Seed: %s\n" % colab.settings['InitialSeed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                write_general_settings(f)
            elif mode == "img2img":
                f.write("Mode: Image to Image\n")
                f.write("Initial Seed: %s\n" % colab.settings['InitialSeed'])
                f.write("Prompt: %s\n" % colab.settings['Prompt'])
                f.write("Negative Prompt: %s\n" % colab.settings['NegativePrompt'])
                f.write("Strength: %s\n" % colab.settings['Strength'])
                f.write("Initial Image URL: %s\n" % colab.settings['InitialImageURL'])
                write_general_settings(f)
            elif mode == "inpaint":
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


def start_post_processing(img, imageName, image_uid, gdrive, replacePreview):
    if gdrive:
        path = save_gdrive(img, imageName)
        display("Saved: %s" % path, display_id=image_uid + "_saved")
    imgSavePath = get_save_path(imageName)
    if colab.settings['Scale'] != "1x":
        import upscaler
        importlib.reload(upscaler)
        img.save("tmp_input.png")
        scale = int(colab.settings['Scale'][:-1])
        scaled_image = upscaler.upscale(colab.settings['Upscaler'], scale, "tmp_input.png")
        if gdrive:
            path = save_gdrive(scaled_image, imageName + "-%dx" % scale)
            # if '/content/gdrive/MyDrive/path exists, then the image was saved to gdrive
            if os.path.exists("/content/gdrive/MyDrive/" + path):
                display("Saved: %s" % path, display_id=image_uid + "_saved")
        else:
            scaled_image.save(imgSavePath + "-%dx.png" % scale)
        # downscale the image to 1x for display
        downscaled_image = scaled_image.resize((img.width, img.height), PIL.Image.LANCZOS)
        if replacePreview:
            display(downscaled_image, display_id=image_uid)
        else:
            display(downscaled_image, display_id=image_uid + ("-%dx" % scale))
        # Save the 2x image in media-dir
        scaled_image.save("media-dir/%s-%dx.png" %(imageName, scale))
        # dispaly the 2x image as a link
        html_link = HTML("<a href='%s%s-%dx.png' target='_blank'>Full %dx-scaled Image</a>" % (colab.server_url, imageName, scale, scale))
        display("Scaled: ", html_link, display_id=image_uid + "_scaled")
post_process_jobs = []
import threading
def job_queue():
    while True:
        if len(post_process_jobs) > 0:
            start_post_processing(*post_process_jobs[0])
            post_process_jobs.pop(0)
        else:
            time.sleep(0.1)
def post_process(img, imageName, image_uid, gdrive = True, replacePreview = True):
    post_process_jobs.append((img, imageName, image_uid, gdrive, replacePreview))


threading.Thread(target=job_queue).start()
