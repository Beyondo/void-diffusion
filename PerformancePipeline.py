# PerformancePipeline Or PP for convenience (Hahah- ok, I'm sorry)
from diffusers import StableDiffusionPipeline
import torch, os, importlib
from hax import safety_patcher, clip_pipeline, clip_limit
importlib.reload(safety_patcher)
importlib.reload(clip_pipeline)
def from_pretrained(model_name):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512"
    torch.set_default_dtype(torch.float16)
    #VOIDPipeline.Hook()
    rev = "diffusers-115k" if  model_name == "naclbit/trinart_stable_diffusion_v2" else "fp16"
    pipe = None
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, revision=rev, torch_dtype=torch.float16, safety_checker=None)
    except:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, safety_checker=None)
        except Exception as e:
            print("Failed to load model %s: %s" % (model_name, e))
    #pipe = clip_limit.modify(512)
    pipe.to("cuda:0")
    return pipe