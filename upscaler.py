import os, PIL.Image, IPython.display, IPython, hashlib
def upscale(upscaler, scale, image_input_path):
    image = PIL.Image.open(image_input_path)
    if upscaler.lower() == "bicubic":
        image = image.resize((image.width * scale, image.height * scale), PIL.Image.BICUBIC)
    elif upscaler.lower() == "gfpgan+realesrgan":
        hash = hashlib.sha256(image.tobytes()).hexdigest()
        temp_dir = "temp/%s" % hash
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(input_dir, "image.png"))
        if os.path.exists(os.path.join(output_dir, "restored_imgs", "image.png")): os.remove(os.path.join(output_dir, "restored_imgs", "image.png"))
        IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i %s -o %s -v 1.3 -s %s --bg_upsampler realesrgan &> /dev/null" % (input_dir, output_dir, scale))
    if not os.path.exists(os.path.join(output_dir, "restored_imgs", "image.png")):
        raise Exception("Failed to upscale image using %s" % upscaler)
    image = PIL.Image.open("temp/output/restored_imgs/image.png")
    return image