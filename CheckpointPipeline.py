from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
import torch
def from_pretrained(checkpoint_path, is_img2img=False):
    torch.set_default_dtype(torch.float16)
    pipe = None
    try:
        pipe = pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
        checkpoint_path,
        device='cuda',
        controlnet=None
    )
    except Exception as e:
        print("Failed to load checkpoint %s: %s" % (checkpoint_path, e))
    return pipe