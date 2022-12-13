import torch, colab, time
from IPython.display import display
last_image_time = 0
def callback(iter, t, latents):
    if time.time() - last_image_time > 3:
        colab.last_image_time = time.time()
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = colab.text2img.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images = colab.text2img.numpy_to_pil(images)
            display(images[0], display_id=colab.get_current_image_uid())