import torch, time
from legacy import colab
from IPython.display import display, HTML
import PIL.Image as Image
import threading
rendering_start_time = 0
last_image_time = 0
replace_result = True
def reset():
    global rendering_start_time
    rendering_start_time = time.time()

def show(img = None, iter = 0):
    global rendering_start_time
    image_id = colab.get_current_image_uid()
    seed = colab.get_current_image_seed()
    seed_label = f'<label>Seed: <code id="seed">{seed}</code></label>'
    copy_button = '<button onclick="copySeed()">Copy</button>'
    javascript = f'''
        function copySeed() {{
            var seed = document.getElementById("seed");
            var range = document.createRange();
            range.selectNode(seed);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand("copy");
            window.getSelection().removeAllRanges();
        }}
    '''
    display(HTML(seed_label + copy_button), display_id=image_id + "_seed")
    display(HTML("<label>Generation time: %.2fs</label>" % (time.time() - rendering_start_time)), display_id=image_id + "_time")
    display(HTML("<label>Original: Emerging...</label>"), display_id=image_id + "_original")
    display(HTML("<label>Saved: No</label>"), display_id=image_id + "_saved")
    display(HTML("<label>Post-processed: Waiting...</label>"), display_id=image_id + "_scaled")
    display(HTML("<label>Saved: No</label>"), display_id=image_id + "_scaled_saved")
    if not img == None:
        display(img, display_id=image_id)
        if not replace_result: display("[Post-processed Image...]", display_id=image_id + "_image_scaled")
    else:
        display("...", display_id=image_id)
    display(Javascript(javascript))

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
            threading.Thread(target=show, args=(images[0], iter)).start()