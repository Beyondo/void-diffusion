import torch
model_name = ""
ready = False
pipe = None
def init(ModelName):
    global model_name, ready, pipe
    model_name = ModelName
    if not torch.cuda.is_available():
        print("No GPU found. Go to Runtime -> Change runtime type, and choose \"GPU\" then click Save.")
    else:
        print("Running on -> ", end="")
        print(torch.cuda.get_device_name("cuda:0") + ".")
        print("Initializing model -> " + model_name + ":")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_name, revision="fp16", torch_dtype=torch.float16)
        print("Done.")
        ready = True
        from IPython.display import clear_output; clear_output()