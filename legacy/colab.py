import torch, random, time
import IPython
from IPython import display
from IPython.display import HTML
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import env, PerformancePipeline, importlib
importlib.reload(PerformancePipeline)
model_name = ""
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
def get_current_image_seed():
    global settings, image_id
    return settings['InitialSeed'] + image_id
def get_current_image_uid():
    return "text2img-%d" % get_current_image_seed()

def media_server():
    global server_url
    # get colab server url
    from google.colab.output import eval_js
    server_url = eval_js("google.colab.kernel.proxyPort(8000)")
    IPython.get_ipython().system_raw("python -m http.server 8000 &")
def init(ModelName, debug=False):
    global model_name, ready, pipeline, tokenizer, img2img, inpaint
    ready = False
    model_name = ModelName
    settings['ModelName'] = ModelName
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Starting local media server ->", end="")
        from threading import Thread
        Thread(target=media_server).start()
        print("Done.\nRunning on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        try:
            env.install_vendor()
            print("Initializing model " + model_name + ":")
            pipeline = PerformancePipeline.from_pretrained(model_name)
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            inpaint = StableDiffusionInpaintPipeline(**pipeline.components)
            print("Done.")
            ready = True
            if not debug:
                from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected." % model_name))
            display.display(HTML("Media server: <a href='%s' target='_blank'>%s</a>" % (server_url, server_url)))
        except Exception as e:
            if "502" in str(e):
                print("Received 502 Server Error: Huggingface is currently down." % model_name)
            print(e)

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