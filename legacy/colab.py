import torch, random, time, os
import IPython
from IPython import display
from IPython.display import HTML
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import env, PerformancePipeline, importlib
from hax import safety_patcher
importlib.reload(PerformancePipeline)
model_name = ""
inpaint_model_name = ""
ready = False
tokenizer = None
pipeline = None
img2img = None
inpaint = None
settings = { }
save_directory = "AI-Gen"
save_settings = True
image_id = 0
current_mode = ""
server_url = ""
last_generated_image = None
image_size = (512, 512)
def get_current_image_seed():
    global settings, image_id
    return settings['InitialSeed'] + image_id
def get_current_image_uid():
    return "%s-%d" % (current_mode, get_current_image_seed())

def media_server():
    global server_url
    from google.colab.output import eval_js
    server_url = eval_js("google.colab.kernel.proxyPort(8000)")
    IPython.get_ipython().system_raw("fuser -k 8000/tcp")
    IPython.get_ipython().system_raw("python -m http.server 8000 --directory media-dir")
def init(ModelName, InpaintingModel, debug=False):
    global model_name, ready, pipeline, tokenizer, img2img, inpaint, settings, server_url
    ready = False
    model_name = ModelName
    inpaint_model_name = InpaintingModel
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        torch.set_default_dtype(torch.float16)
        print(torch.cuda.get_device_name("cuda:0") + ".")
        try:
            env.install_vendor()
            print("Initializing model " + model_name + ":")
            pipeline = PerformancePipeline.from_pretrained(model_name)
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            if InpaintingModel != None:
                try:
                    inpaint = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model_name, revision="fp16", torch_dtype=torch.float16).to("cuda:0")
                    safety_patcher.try_patch(inpaint)
                except:
                    try:
                        inpaint = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model_name, torch_dtype=torch.float16).to("cuda:0")
                        safety_patcher.try_patch(inpaint)
                    except:
                        print("Couldn't load %s as an Inpainting model." % inpaint_model_name)
                        return
            safety_patcher.try_patch(pipeline)
            safety_patcher.try_patch(img2img)
            print("Done.")
            ready = True
            if not debug:
                from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected for Text2Img and Img2Img." % model_name))
            if InpaintingModel == None:
                display.display(HTML("Inpainting model <strong><span style='color: red'>not selected</span></strong>."))
            else:
                display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected for inpainting." % inpaint_model_name))
        except Exception as e:
            if "502" in str(e):
                print("Received 502 Server Error: Huggingface is currently down." % model_name)
            print(e)
def start_media_server():
    from threading import Thread
    Thread(target=media_server).start()
def prepare(mode):
    start_media_server()
    global current_mode, settings
    torch.set_default_dtype(torch.float16)
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