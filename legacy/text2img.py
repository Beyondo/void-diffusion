import torch, os, time, datetime, importlib
from legacy import colab, postprocessor, progress
from IPython.display import display
importlib.reload(progress)
importlib.reload(postprocessor)
def process(ShouldSave, maxNumJobs, ShouldPreview = True, ReplaceResult = True):
    progress.replace_result = ReplaceResult
    colab.prepare("text2img")
    timestamp = int(time.mktime(datetime.datetime.now().timetuple()))
    if ShouldSave and colab.save_settings: postprocessor.save_settings(timestamp, mode="text2img")
    num_iterations = colab.settings['Iterations']
    display("Iterations: 0/%d" % num_iterations, display_id="iterations")
    postprocessor.run_queue_thread()
    for i in range(num_iterations):
        colab.image_id = i # needed for progress.py
        generator = torch.Generator("cuda").manual_seed(colab.settings['InitialSeed'] + i)
        progress.reset()
        progress.show()
        torch.cuda.empty_cache()
        image = colab.pipeline(
            prompt=colab.settings['Prompt'],
            width=colab.settings['Width'],
            height=colab.settings['Height'],
            negative_prompt=colab.settings['NegativePrompt'],
            guidance_scale=colab.settings['GuidanceScale'],
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