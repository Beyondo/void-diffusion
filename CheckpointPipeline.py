# PerformancePipeline Or PP for convenience (Hahah- ok, I'm sorry)
from diffusers import StableDiffusionPipeline
import torch, os, importlib
from hax import clip_pipeline, clip_limit
importlib.reload(clip_pipeline)
def from_pretrained(checkpoint_path):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512"
    torch.set_default_dtype(torch.float16)
    #VOIDPipeline.Hook()
    pipe = None
    try:
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16, safety_checker=None)
    except Exception as e:
        print("Failed to load model %s: %s" % (checkpoint_path, e))
    #pipe = clip_limit.modify(512)
    pipe.to("cuda:0")
    return pipe