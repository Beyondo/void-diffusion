import torch, time
from legacy import colab
from IPython.display import display
rendering_start_time = 0
last_image_time = 0
def reset():
    global rendering_start_time
    rendering_start_time = time.time()

def show(img = None):
    global rendering_start_time
    display("Seed: %d" % colab.get_current_image_seed(), display_id=colab.get_current_image_uid() + "_seed")
    display("Execution time: %.2fs" % (time.time() - rendering_start_time), display_id=colab.get_current_image_uid() + "_time")
    display("Scaled: No.", display_id=colab.get_current_image_uid() + "_scaled")
    display("Saved: No.", display_id=colab.get_current_image_uid() + "_saved")
    if not img == None: display(img, display_id=colab.get_current_image_uid())
    else: display("...", display_id=colab.get_current_image_uid())
def callback(iter, t, latents):
    global last_image_time
    if time.time() - last_image_time > 3:
        last_image_time = time.time()
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = colab.pipeline.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images = colab.pipeline.numpy_to_pil(images)
            show(images[0])