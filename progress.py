import torch, colab
from IPython.display import display
def callback(iter, t, latents):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = colab.text2img.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = colab.text2img.numpy_to_pil(image)
        for i, img in enumerate(image):
            display(img, display_id=str(i))