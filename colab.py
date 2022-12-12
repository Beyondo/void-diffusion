import patcher, torch, random, time
model_name = ""
ready = False
tokenizer = None
text2img = None
img2img = None
inpaint = None
settings = { }
save_directory = "AI-Gen"
save_settings = True
image_id = 0
current_mode = ""
def get_current_image_seed():
    global settings, image_id
    return settings['InitialSeed'] + image_id
def get_current_image_uid():
    return "text2img-%d" % get_current_image_seed()
def init(ModelName):
    global model_name, ready, text2img, img2img, inpaint
    model_name = ModelName
    settings['ModelName'] = ModelName
    patcher.patch()
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        print("Initializing model -> " + model_name + ":")
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
        from transformers import AutoTokenizer
        text2img = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16", torch_dtype=torch.float16).to("cuda:0")
        img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        print("Done.")
        ready = True
        from IPython.display import clear_output; clear_output()
        print("Model '" + model_name + "' has been selected.")

def prepare(mode):
    global current_mode, settings
    if 'Seed' not in settings:
        print("Please set your settings first.")
        return
    if settings['Seed'] == 0:
        random.seed(int(time.time_ns()))
        settings['InitialSeed'] = random.getrandbits(64)
    else:
        settings['InitialSeed'] = settings['Seed']
    current_mode = mode