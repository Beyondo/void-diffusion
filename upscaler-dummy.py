try:
    import os, PIL.Image, IPython.display, IPython, hashlib
except: print("Upscaling error: %s" % e)
def bicubic(image, scale):
    return image.resize((image.width * scale, image.height * scale), PIL.Image.BICUBIC)
def nearest(image, scale):
    return image.resize((image.width * scale, image.height * scale), PIL.Image.NEAREST)
def gfpgan(image, scale, bg_sampler = None):
    hash = hashlib.sha256(image.tobytes()).hexdigest()
    temp_dir = "temp/%s" % hash
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(input_dir, "image.png"))
    IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i %s -o %s -v 1.3 -s %s %s &> /dev/null" % (input_dir, output_dir, scale, "--bg_upsampler %s" % bg_sampler if bg_sampler else ""))
    try:
        image = PIL.Image.open(os.path.join(output_dir, "restored_imgs", "image.png"))
    except Exception as e: print("Scaling failed: %s" % e)
    return image
def gfpgan_all(scale, bg_sampler = None):
    input_dir = "temp/input"
    output_dir = "temp/output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i %s -o %s -v 1.3 -s %s %s &> /dev/null" % (input_dir, output_dir, scale, "--bg_upsampler %s" % bg_sampler if bg_sampler else ""))
    return True
def realesrgan(image, scale):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load("vendor/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x2plus.pth"), strict=True)
    model.eval()
    model = model.to("cuda:0")
def esrgan(image, scale):
    import cv2, torch
    import vendor.ESRGAN.RRDBNet_arch as arch
    img = cv2.cvtColor(np.array(bicubic(image, scale)), cv2.COLOR_RGB2BGR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to("cuda:0")
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = output.astype(np.uint8)
    return PIL.Image.fromarray(output)
upscalers = { }
upscalers['bicubic'] = [lambda image, scale: bicubic(image, scale), lambda scale: bicubic(PIL.Image.open("temp/input/image.png"), scale)]
upscalers['nearest'] = [lambda image, scale: nearest(image, scale), lambda scale: nearest(PIL.Image.open("temp/input/image.png"), scale)]
upscalers['gfpgan'] = [lambda image, scale: gfpgan(image, scale), lambda scale: gfpgan_all(scale)]
upscalers['esrgan'] =  [lambda image, scale: esrgan(image, scale), lambda scale: esrgan(PIL.Image.open("temp/input/image.png"), scale)]
upscalers['gfpgan+real-esrgan'] = [lambda image, scale: gfpgan(image, scale, "real-esrgan"), lambda scale: gfpgan_all(scale, "real-esrgan")]
def upscale(upscaler, scale, image):
    image = upscalers[upscaler.lower()][0](image, scale)
    return image
def upscale_all(upscaler, scale):
    upscalers[upscaler.lower()][1](scale)