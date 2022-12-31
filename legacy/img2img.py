import torch, os, time, datetime, importlib
from legacy import colab, postprocessor, progress
from IPython.display import Image
from IPython.display import display

import requests
from PIL import Image
from io import BytesIO
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, maxNumJobs, ShouldPreview = True, ReplaceResult = True):
    try:
        progress.replace_result = ReplaceResult
        colab.prepare("img2img")
        timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
        if colab.save_settings: postprocessor.save_settings(timestamp, mode="img2img")
        # Load image
        init_image = None
        if colab.settings['UseLastOutputAsInitialImage'] and colab.last_generated_image is not None:
            init_image = colab.last_generated_image
        else:
            init_image = Image.open(BytesIO(requests.get(colab.settings['InitialImageURL']).content)).convert('RGB')
        init_image.thumbnail((colab.settings['Width'], colab.settings['Height']))
        display(init_image)
        # Process image
        num_iterations = colab.settings['Iterations']
        display("Iterations: 0/%d" % num_iterations, display_id="iterations")
        postprocessor.max_num_parallel_jobs = maxNumJobs
        postprocessor.run_queue_thread()
        for i in range(num_iterations):
            colab.image_id = i # needed for progress.py
            colab.current_seed = colab.settings['InitialSeed'] + i
            generator = torch.Generator("cuda").manual_seed(colab.current_seed)
            progress.reset()
            progress.show()
            torch.cuda.empty_cache()
            image = colab.img2img(
                prompt=colab.settings['Prompt'],
                image=init_image,
                negative_prompt=colab.settings['NegativePrompt'],
                guidance_scale=colab.settings['GuidanceScale'],
                strength=colab.settings['Strength'],
                num_inference_steps=colab.settings['Steps'],
                generator=generator,
                callback=progress.callback if ShouldPreview else None,
                callback_steps=20).images[0]
            colab.last_generated_image = image
            progress.show(image)
            postprocessor.post_process(image, "%d_%d" % (timestamp, i), colab.get_current_image_uid(), ShouldSave, ReplaceResult)
            display("Iterations: %d/%d" % (i + 1,  num_iterations), display_id="iterations")
        postprocessor.join_queue_thread()
        torch.cuda.empty_cache()
    except Exception as e:
        print("Error trying to generate image: " + str(e))