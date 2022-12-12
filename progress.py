import torch, colab
from IPython.display import display
def callback(iter, t, latents):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        images = colab.text2img.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = colab.text2img.numpy_to_pil(images)