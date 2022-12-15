import patcher, torch, random, time
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
def init(ModelName):
    global model_name, ready, pipeline, tokenizer, text2img, img2img, inpaint
    ready = False
    model_name = ModelName
    settings['ModelName'] = ModelName
    patcher.patch()
    if not torch.cuda.is_available():
        print("No GPU found. If you are on Colab, go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        try:
            rev = "diffusers-115k" if model_name == "naclbit/trinart_stable_diffusion_v2" else "fp16"
            print("-> Initializing model " + model_name + ":")
            #import VOIDPipeline
            #import importlib
            #importlib.reload(VOIDPipeline)
            #VOIDPipeline.Take_Over()
            
            # Why does it generate an image that has nothing to do with the text?
            # -> Because the text encoder is not trained on the same dataset as the image encoder.
            torch.set_default_dtype(torch.float16)
            # get clip text config
            config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
            config.atttention_dropout = 0.0
            config.bos_token_id = 0
            config.dropout = 0.0
            config.eos_token_id = 2
            config.hidden_act = "quick_gelu"
            config.hidden_size = 768
            config.initializer_factor = 1.0
            config.initializer_range = 0.02
            config.intermediate_size = 3072
            config.layer_norm_eps = 1e-05
            config.max_position_embeddings = 512
            config.model_type = "clip_text_model"
            config.num_attention_heads = 12
            config.num_hidden_layers = 12
            config.pad_token_id = 1
            config.projection_dim = 768
            config.torch_dtype = "float32"
            config.transformers_version = "4.22.0.dev0"
            config.vocab_size = 49408


            pipeline = StableDiffusionPipeline.from_pretrained(model_name, revision=rev).to("cuda:0")
            pipeline.text_encoder = CLIPTextModel(config).to("cuda:0")
            pipeline.tokenizer.model_max_length = 512
            pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
            text2img = StableDiffusionPipeline(**pipeline.components)
            img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
            inpaint = StableDiffusionInpaintPipeline(**pipeline.components)
            print("Done.")
            ready = True
            #from IPython.display import clear_output; clear_output()
            display.display(HTML("Model <strong><span style='color: green'>%s</span></strong> has been selected." % model_name))
        except Exception as e:
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