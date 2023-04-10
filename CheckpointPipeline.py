from convert_ckpt_to_diffusers import load_pipeline_from_original_stable_diffusion_ckpt

import torch
def from_pretrained(checkpoint_path, is_img2img=False):
    torch.set_default_dtype(torch.float32)
    pipe = None
    try:
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(
            checkpoint_path,
            controlnet=None,
            precision=torch.float16,
            scan_needed=False
        )
        pipe.to('cuda')
    except Exception as e:
        print("Failed to load checkpoint %s: %s" % (checkpoint_path, e))
    return pipe