import os, PIL.Image, IPython.display, IPython, hashlib
def gfpgan(image, scale, bg_sampler = None):
    hash = hashlib.sha256(image.tobytes()).hexdigest()
    temp_dir = "temp/%s" % hash
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(input_dir, "image.png"))
    if os.path.exists(os.path.join(output_dir, "restored_imgs", "image.png")): os.remove(os.path.join(output_dir, "restored_imgs", "image.png"))
    IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i %s -o %s -v 1.3 -s %s %s &> /dev/null" % (input_dir, output_dir, scale, "--bg_upsampler %s" % bg_sampler if bg_sampler else ""))
    if not os.path.exists(os.path.join(output_dir, "restored_imgs", "image.png")):
        raise Exception("Failed to upscale image using %s" % upscaler)
    image = PIL.Image.open(os.path.join(output_dir, "restored_imgs", "image.png"))
    if os.path.exists(temp_dir): os.system("rm -rf %s" % temp_dir)
    return image

upscalers = { }
upscalers['bicubic'] = lambda image, scale: image.resize((image.width * scale, image.height * scale), PIL.Image.BICUBIC)
upscalers['nearest'] = lambda image, scale: image.resize((image.width * scale, image.height * scale), PIL.Image.NEAREST)
upscalers['gfpgan'] = lambda image, scale: gfpgan(image, scale)
upscalers['gfpgan_realesrgan'] = lambda image, scale: gfpgan(image, scale, bg_sampler = "realesrgan")
def upscale(upscaler, scale, image_input_path):
    image = PIL.Image.open(image_input_path)
    image = upscalers[upscaler.lower()](image, scale)
    return image