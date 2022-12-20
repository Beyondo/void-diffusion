
def replace_everything(target_dir, from_text, to_text):
    global pipeline, model_name
    (repository_id, name) = model_name.split("/")
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    data = f.read()
                data = data.replace(from_text, to_text)
                with open(path, "w") as f:
                    f.write(data)
                    print("Wrote to " + path)
def modify(limit):
    global pipeline, model_name
    (repository_id, name) = model_name.split("/")
    replace_everything("/root/.cache/huggingface/diffusers/models--%s--%s/snapshots/" % (repository_id, name), "model_max_length\": 77", "model_max_length\": %d" % limit)
    replace_everything("/root/.cache/huggingface/diffusers/models--%s--%s/snapshots/" % (repository_id, name), "max_position_embeddings\": 77", "max_position_embeddings\": %d" % limit)
    from shutil import copytree, ignore_patterns
    copytree("/root/.cache/huggingface/diffusers/models--%s--%s/snapshots/" % (repository_id, name), "/content/void-diffusion/stable-diffusion-1.5v")
    pipeline =  StableDiffusionPipeline.from_pretrained("/content/void-diffusion/stable-diffusion-1.5v/ded79e214aa69e42c24d3f5ac14b76d568679cc2", revision="fp16", torch_dtype=torch.float16).to("cuda:0")
    old_weights = pipeline.text_encoder.text_model.embeddings.position_embedding.weight.data.to("cuda:0")
    input_embeddings = pipeline.text_encoder.text_model.embeddings.token_embedding
    pipeline.text_encoder.config.max_position_embeddings = limit
    # Bug: The following line is supposed to be a hack to make the model reload everything using the new config but it also makes the model generate random images:
    #pipeline.text_encoder.text_model.__init__(config=pipeline.text_encoder.config)
    # Which might be because the model wasn't trained to receive N number of tokens to begin with,
    # however, that might not be the case since if I tried with the default value, that's "77" and uncommenting that line, it still generates random images.
    # So there's still the possibility that there might be a way to make it work, but I don't know how.
    # In any case, it's not as trivial as I thought.
    pipeline.text_encoder.text_model.to("cuda:0")
    pipeline.text_encoder.text_model.embeddings.token_embedding = input_embeddings
    pipeline.text_encoder.text_model.embeddings.position_embedding = torch.nn.Embedding(limit, 768).to("cuda:0") # Zero padding
    pipeline.text_encoder.text_model.embeddings.position_embedding.weight.data[:old_weights.shape[0]] = old_weights
    # Tokenizer
    #pipeline.tokenizer.model_max_length = limit
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    # TODO: Add a custom denoiser that uses the new limit
    return pipeline