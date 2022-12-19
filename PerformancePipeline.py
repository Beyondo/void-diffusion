# Or PP for convenience (Hahah- ok, I'm sorry)
from diffusers import StableDiffusionPipeline
import torch, os, importlib
from hax import safety_patcher, clip_pipeline, clip_limit
importlib.reload(safety_patcher)
importlib.reload(clip_pipeline)
fp16_models = ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]
def from_pretrained(name):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512"
    torch.set_default_dtype(torch.float16)
    #VOIDPipeline.Hook()
    rev = "diffusers-115k" if  model_name == "naclbit/trinart_stable_diffusion_v2" else "fp16" if model_name in fp16_models else ""
    # Todo: add revision detection for any model using huggingface's model hub
    if(rev != ""):
        pipe = StableDiffusionPipeline.from_pretrained(name, revision=rev, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(name, torch_dtype=torch.float16)
    #pipe = clip_limit.modify(512)
    pipe.to("cuda:0")
    safety_patcher.try_patch(pipe)
    return model