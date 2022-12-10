import subprocess
def install(name):
    subprocess.call(['pip', 'install', name])
install("torch")
import torch
def init():
    if not torch.cuda.is_available():
        print("No GPU found. Go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0"), end=".\nInstalling dependencies -> ")
        install("einops torch transformers diffusers accelerate > /dev/null")
        print("Done.\nUsing model -> ", end="")
        Model = "CompVis/stable-diffusion-v1-4" # @param ["CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion"]
        # @markdown **Available models**:<br>
        # @markdown `CompVis/stable-diffusion-v1-4` -> Trained on everything<br>
        # @markdown `hakurei/waifu-diffusion` -> Trained on anime<br>
        from diffusers import StableDiffusionPipeline
        import sys
        sys.stdout = open('stdout.txt', 'w')
        pipe = StableDiffusionPipeline.from_pretrained(Model, revision="fp16", torch_dtype=torch.float16)
        print(Model, end=".\n")