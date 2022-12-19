import torch, random, time
from IPython import display
from IPython.display import HTML
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import env, PerformancePipeline
model_name = ""
ready = False
tokenizer = None
pipeline = None
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
    
def init(ModelName, debug=False):
    global model_name, ready, pipeline, tokenizer, text2img, img2img, inpaint
    ready = False
    model_name = ModelName
    settings['ModelName'] = ModelName
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        try:
            env.install_vendor()
            print("Initializing model " + model_name + ":")
            text2img = PerformancePipeline.from_pretrained(model_name)
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            inpaint = StableDiffusionInpaintPipeline(**pipeline.components)
            print("Done.")
            ready = True
            if not debug:
                from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected." % model_name))
        except Exception as e:
            if "502" in str(e):
                print("Received 502 Server Error: Huggingface is currently down." % model_name)
            print("Failed to initialize model %s with error %s" % (model_name, e))

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
    torch.cuda.empty_cache()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    import PIL.Image
    w, h = imgs[0].size
    grid = PIL.Image.new('RGBA', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        if img.mode != 'RGBA': img = img.convert('RGBA')
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid