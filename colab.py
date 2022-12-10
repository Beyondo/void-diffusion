import patcher, torch
model_name = ""
ready = False
text2img = None
img2img = None
inpaint = None
def init(ModelName):
    global model_name, ready, text2img, img2img, inpaint
    model_name = ModelName
    patcher.patch()
    if not torch.cuda.is_available():
        print("No GPU found. Go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        print("Initializing model -> " + model_name + ":")
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
        text2img = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16", torch_dtype=torch.float16).to("cuda:0")
        img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        print("Done.")
        ready = True
        from IPython.display import clear_output; clear_output()