import torch, time
from legacy import colab
from IPython.display import display
from IPython.display import HTML
rendering_start_time = 0
last_image_time = 0
replace_result = True
def reset():
    global rendering_start_time
    rendering_start_time = time.time()

def show(img = None):
    global rendering_start_time
    image_id = colab.get_current_image_uid()
    display(HTML("<label>Seed: <code>%d</code></label>" % colab.get_current_image_seed()), display_id=image_id + "_seed")
    display(HTML("<label>Execution time: %.2fs</label>" % (time.time() - rendering_start_time)), display_id=image_id + "_time")
    display(HTML("<label>Original: Emerging...</label>"), display_id=image_id + "_original")
    display(HTML("<label>Saved: No</label>"), display_id=image_id + "_saved")
    display(HTML("<label>Scaled: Waiting...</label>"), display_id=image_id + "_scaled")
    display(HTML("<label>Saved: No</label>"), display_id=image_id + "_scaled_saved")
    if not img == None:
        display(img, display_id=image_id)
        if not replace_result: display("[Scaled Image...]", display_id=image_id + "_image_scaled")
    else:
        display("...", display_id=image_id)
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
            show(images[0].resize(colab.image_size))