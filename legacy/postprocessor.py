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

def post_process(img, imageName, gdrive = True, replacePreview = True):
    if gdrive:
        path = save_gdrive(img, imageName)
        print("Saved to " + path)
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
                print("Saved to " + path)
        else:
            scaled_image.save(imgSavePath + "-%dx.png" % scale)
        # downscale the image to 1x for display
        downscaled_image = scaled_image.resize((img.width, img.height), PIL.Image.LANCZOS)
        if replacePreview:
            display(downscaled_image, display_id=colab.get_current_image_uid())
        else:
            display(downscaled_image, display_id=colab.get_current_image_uid() + ("-%dx" % scale))
        blob = scaled_image.tobytes()
        blob = base64.b64encode(blob).decode("utf-8")
        display(HTML("<a href='data:image/png;base64,%s' target='_blank'>Full resolution</a>" % (blob, scale)))