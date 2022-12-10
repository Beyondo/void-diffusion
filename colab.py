import torch
def init(model_name):
    if not torch.cuda.is_available():
        print("No GPU found. Go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        print("Initializing model -> " + model_name + ":")
        from diffusers import StableDiffusionPipeline
        import sys
        sys.stdout = open('stdout.txt', 'w')
        pipe = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16", torch_dtype=torch.float16)
        print("Done.")