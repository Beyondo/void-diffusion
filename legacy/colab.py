import torch, random, time, os, gc, diffusers
import IPython
from IPython import display
from IPython.display import HTML
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import PerformancePipeline, CheckpointPipeline, importlib
import traceback
importlib.reload(PerformancePipeline)
importlib.reload(CheckpointPipeline)
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
current_seed = 0
default_pipe_scheduler = None
default_inpaint_scheduler = None
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
def prepare_memory():
    gc.collect()
    torch.cuda.empty_cache()
def init(ModelName, InpaintingModel, debug=False):
    global model_name, ready, pipeline, tokenizer, img2img, inpaint, settings, server_url, default_pipe_scheduler, default_inpaint_scheduler
    prepare_memory()
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
            if ModelName.endswith('.ckpt'):
                print("Initializing checkpoint " + model_name + ":")
                pipeline = CheckpointPipeline.from_pretrained(model_name)
            else:
                print("Initializing model " + model_name + ":")
                pipeline = PerformancePipeline.from_pretrained(model_name)
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            default_pipe_scheduler = pipeline.scheduler
            if InpaintingModel != None:
                try:
                    if InpaintingModel.endswith('.ckpt'):
                        inpaint = CheckpointPipeline.from_pretrained(model_name, is_img2img=True)
                    else:
                        inpaint = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model_name, revision="fp16", torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
                except:
                    try:
                        if InpaintingModel.endswith('.ckpt'):
                            inpaint = CheckpointPipeline.from_pretrained(model_name, is_img2img=True)
                        else:
                            inpaint = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model_name, torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
                    except:
                        print("Couldn't load %s as an Inpainting model." % inpaint_model_name)
                        return
            print("Done.")
            ready = True
            if not debug:
                from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected for Text2Img and Img2Img." % model_name))
            if InpaintingModel == None:
                display.display(HTML("Inpainting model <strong><span style='color: red'>not selected</span></strong>."))
            else:
                display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected for inpainting." % inpaint_model_name))
                default_inpaint_scheduler = inpaint.scheduler
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
    try:
        if settings['Seed'] == 0:
            random.seed(int(time.time_ns()))
            settings['InitialSeed'] = random.getrandbits(64)
        else:
            settings['InitialSeed'] = settings['Seed']
        current_mode = mode
        if mode == "text2img" or mode == "img2img":
            if settings['Scheduler'] != "Default":
                scheduler = getattr(diffusers, settings['Scheduler'])
                pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
            else:
                pipeline.scheduler = default_pipe_scheduler
        elif mode == "inpaint":
            if settings['Scheduler'] != "Default":
                scheduler = getattr(diffusers, settings['Scheduler'])
                inpaint.scheduler = scheduler.from_config(inpaint.scheduler.config)
            else:
                inpaint.scheduler = default_inpaint_scheduler
        prepare_memory()
    except Exception:
        print(traceback.format_exc())
#
def image_grid(imgs, rows, cols):#
    assert len(imgs) == rows*cols
    import PIL.Image
    w, h = imgs[0].size
    grid = PIL.Image.new('RGBA', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        if img.mode != 'RGBA': img = img.convert('RGBA')
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid