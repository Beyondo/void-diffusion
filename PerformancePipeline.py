# PerformancePipeline Or PP for convenience (Hahah- ok, I'm sorry)
from diffusers import StableDiffusionPipeline
import torch

def from_pretrained(model_name, safety_checker=None):
    torch.set_default_dtype(torch.float16)
    rev = "diffusers-115k" if  model_name == "naclbit/trinart_stable_diffusion_v2" else "" if model_name == "SG161222/Realistic_Vision_V2.0" else "fp16"
    pipe = None
    try:
        if rev != "":
            pipe = StableDiffusionPipeline.from_pretrained(model_name, revision=rev, torch_dtype=torch.float16, safety_checker=safety_checker)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, safety_checker=safety_checker)
        pipe.to("cuda")
    except:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, safety_checker=safety_checker)
            pipe.to("cuda")
        except Exception as e:
            print("Failed to load model %s: %s" % (model_name, e))
    return pipe