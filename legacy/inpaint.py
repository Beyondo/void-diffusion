import torch, os, time, datetime, importlib
from legacy import colab, postprocessor, progress
from IPython.display import Image, HTML, display

import requests, traceback
from PIL import Image
from io import BytesIO
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, maxNumJobs, ShouldPreview = True, ReplaceResult = True):
    try:
        progress.replace_result = ReplaceResult
        colab.prepare("inpaint")
        # Since Inpainting is always a Width and Height of 512 - Needed for config saving. 
        colab.settings['Width'] = 512
        colab.settings['Height'] = 512
        #
        timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
        if colab.save_settings: postprocessor.save_settings(timestamp, mode="inpaint")
        # Load image
        init_image = None
        if colab.settings['UseLastOutputAsInitialImage'] and colab.last_generated_image is not None:
            init_image = colab.last_generated_image
        else:
            init_image = Image.open(BytesIO(requests.get(colab.settings['InitialImageURL']).content)).convert('RGB')
        colab.image_size = init_image.size
        mask_image = Image.open(BytesIO(requests.get(colab.settings['MaskImageURL']).content)).convert("RGB")
        if mask_image.size[0] / mask_image.size[1] != init_image.size[0] / init_image.size[1]:
            display("Warning: Mask aspect ratio is different from image aspect ratio. This will cause unexpected results.")
        if mask_image.size[0] < init_image.size[0] or mask_image.size[1] < init_image.size[1]:
            mask_image = mask_image.resize(init_image.size)
        elif mask_image.size[0] > init_image.size[0] or mask_image.size[1] > init_image.size[1]:
            mask_image = mask_image.resize(init_image.size, resample=Image.LANCZOS)
        # Display
        init_image.thumbnail((512, 512))
        mask_image.thumbnail((512, 512))
        mask_applied_image = Image.blend(init_image, mask_image, 0.5)
        display(colab.image_grid([init_image, mask_image, mask_applied_image], 1, 3))
        # Process image
        grey_mask = mask_image.convert("L")
        init_image = init_image.resize((512, 512))
        grey_mask = grey_mask.resize((512, 512))
        num_iterations = colab.settings['Iterations']
        display(HTML("<label>Iterations: 0/%d</label>" % num_iterations), display_id="iterations")
        postprocessor.max_num_parallel_jobs = maxNumJobs
        postprocessor.run_queue_thread()
        for i in range(num_iterations):
            colab.image_id = i # needed for progress.py
            colab.current_seed = colab.settings['InitialSeed'] + i
            generator = torch.Generator("cuda").manual_seed(colab.current_seed)
            progress.reset()
            progress.show()
            latents = None
            if False:
                # generate random image latents for inpainting
                latents = torch.randn(1, 4, 64, 64, device="cuda")
                # blend the mask into the latents
                latents = latents * (1 - mask_image.convert("L").resize((64, 64), Image.BILINEAR).convert("RGB"))
            torch.cuda.empty_cache()
            image = colab.inpaint(
                prompt=colab.settings['Prompt'],
                image=init_image,
                mask_image=grey_mask,
                negative_prompt=colab.settings['NegativePrompt'],
                guidance_scale=colab.settings['GuidanceScale'],
                num_inference_steps=colab.settings['Steps'],
                generator=generator,
                callback=progress.callback if ShouldPreview else None,
                callback_steps=10).images[0]
            image = image.resize((512, int(512 * colab.image_size[1] / colab.image_size[0])))
            image.thumbnail((512, 512))
            colab.last_generated_image = image
            progress.show(image)
            postprocessor.post_process(image, f"{timestamp}_{i}_{colab.current_seed}", colab.get_current_image_uid(), ShouldSave, ReplaceResult)
            display(HTML("<label>Iterations: %d/%d</label>" % (i + 1,  num_iterations)), display_id="iterations")
        postprocessor.join_queue_thread()
        torch.cuda.empty_cache()
    except Exception:
        print("Error trying to generate image", traceback.format_exc())
