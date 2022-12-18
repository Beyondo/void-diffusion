import patcher, torch, random, time, importlib, os
importlib.reload(patcher)
from IPython import display
from IPython.display import HTML
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers.schedulers import PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPTextModel, CLIPTextConfig
import ClipGuided
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
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
# v1.4 = laion/CLIP-ViT-B-32-laion2B-s34B-b79K
# v1.5 = sentence-transformers/clip-ViT-L-14
def create_guided_pipeline(pipeline):
    clip_model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=torch.float16).to("cuda:0")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_name, torch_dtype=torch.float16)
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True)
    guided_pipeline = ClipGuided.CLIPGuidedStableDiffusion(
        unet=pipeline.unet,
        vae=pipeline.vae,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        scheduler=scheduler,
        clip_model=clip_model,
        feature_extractor=feature_extractor,
    )
    return guided_pipeline
def modify_clip_limit(limit):
    # search the entire filestystem for the file tokenizer_config.json
    global pipeline, model_name
    # runwayml--stable-diffusion-v1-5/snapshots/"
    (repository_id, name) = model_name.split("/")
    target_dir = "/root/.cache/huggingface/diffusers/models--%s--%s/snapshots/" % (repository_id, name)
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file == "tokenizer_config.json":
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    data = f.read()
                data = data.replace("77", str(limit))
                with open(path, "w") as f:
                    f.write(data)
                    print("Wrote to " + path)
    # Text Encoder
    old_weights = pipeline.text_encoder.text_model.embeddings.position_embedding.weight.data.to("cuda:0")
    input_embeddings = pipeline.text_encoder.text_model.embeddings.token_embedding
    pipeline.text_encoder.config.max_position_embeddings = limit
    # Bug: The following line is supposed to be a hack to make the model reload everything using the new config but it also makes the model generate random images:
    #pipeline.text_encoder.text_model.__init__(config=pipeline.text_encoder.config)
    # Which might be because the model wasn't trained to receive N number of tokens to begin with,
    # however, that might not be the case since if I tried with the default value, that's "77" and uncommenting that line, it still generates random images.
    # So there's still the possibility that there might be a way to make it work, but I don't know how.
    # In any case, it's not as trivial as I thought.
    pipeline.text_encoder.text_model.to("cuda:0")
    pipeline.text_encoder.text_model.embeddings.token_embedding = input_embeddings
    pipeline.text_encoder.text_model.embeddings.position_embedding = torch.nn.Embedding(limit, 768).to("cuda:0") # Zero padding
    pipeline.text_encoder.text_model.embeddings.position_embedding.weight.data[:old_weights.shape[0]] = old_weights
    # Tokenizer
    pipeline.tokenizer.model_max_length = limit
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    
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
            install_vendor()
            print("Initializing model " + model_name + ":")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512"
            torch.set_default_dtype(torch.float16)
            rev = "diffusers-115k" if model_name == "naclbit/trinart_stable_diffusion_v2" else "" if model_name == "prompthero/openjourney" else "fp16"
            # Hook VOIDPipeline to StableDiffusionPipeline
            import VOIDPipeline, importlib
            importlib.reload(VOIDPipeline)
            VOIDPipeline.Hook()
            if rev != "":
                pipeline = StableDiffusionPipeline.from_pretrained(model_name, revision=rev, torch_dtype=torch.float16).to("cuda:0")
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
            modify_clip_limit(77)
            patcher.patch(pipeline)
            text2img = pipeline
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            inpaint = StableDiffusionInpaintPipeline(**pipeline.components)
            print("Done.")
            ready = True
            if not debug:
                from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected." % model_name))
        except Exception as e:
            # if contains "502 Server Error"
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

def install_vendor():
    print("Installing vendors -> ", end="")
    import os, IPython
    if(os.path.exists("vendor")):
        print("Vendor already installed.")
        return
    try:
        os.mkdir("vendor")
        # GFPGAN
        os.remove("vendor/GFPGAN") if os.path.exists("vendor/GFPGAN") else None
        # git clone using IPython magic
        IPython.get_ipython().system("git clone https://github.com/TencentARC/GFPGAN.git vendor/GFPGAN &> /dev/null")
        IPython.get_ipython().system("pip install basicsr &> /dev/null")
        IPython.get_ipython().system("pip install facexlib &> /dev/null")
        IPython.get_ipython().system("pip install -q -r vendor/GFPGAN/requirements.txt &> /dev/null")
        IPython.get_ipython().system("python vendor/GFPGAN/setup.py develop &> /dev/null")
        # used for enhancing the background (non-face) regions
        IPython.get_ipython().system("pip install realesrgan &> /dev/null")
        # used for enhancing the background (non-face) regions
        IPython.get_ipython().system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -p experiments/pretrained_models &> /dev/null")
        IPython.get_ipython().system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.3.pth -P experiments/pretrained_models &> /dev/null")
        # ESRGAN
        print("Done.")
    except Exception as e:
        print("Error: %s" % e)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    import PIL.Image
    w, h = imgs[0].size
    grid = PIL.Image.new('RGBA', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        if img.mode != 'RGBA': img = img.convert('RGBA')
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid