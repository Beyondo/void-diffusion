import torch, time
from legacy import colab
from IPython.display import display, HTML, Javascript
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
    copy_button = f'<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M10 8V7C10 6.05719 10 5.58579 10.2929 5.29289C10.5858 5 11.0572 5 12 5H17C17.9428 5 18.4142 5 18.7071 5.29289C19 5.58579 19 6.05719 19 7V12C19 12.9428 19 13.4142 18.7071 13.7071C18.4142 14 17.9428 14 17 14H16M7 19H12C12.9428 19 13.4142 19 13.7071 18.7071C14 18.4142 14 17.9428 14 17V12C14 11.0572 14 10.5858 13.7071 10.2929C13.4142 10 12.9428 10 12 10H7C6.05719 10 5.58579 10 5.29289 10.2929C5 10.5858 5 11.0572 5 12V17C5 17.9428 5 18.4142 5.29289 18.7071C5.58579 19 6.05719 19 7 19Z" stroke="#464455" stroke-linecap="round" stroke-linejoin="round"></path></g></svg>'
    javascript = f'''
        function copyToClipboard(text) {{
            var dummy = document.createElement("input");
            document.body.appendChild(dummy);
            dummy.setAttribute('value', text);
            dummy.select();
            document.execCommand('copy');
            document.body.removeChild(dummy);
        }}
    '''
    display(HTML(f'<label>Seed: <code>{seed}</code><button style="width: 16px; padding: 0; margin: 0;" class="copy-button" onclick="copyToClipboard(\'{seed}\')">{copy_button}</button></label>'), display_id=image_id + "_seed")
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