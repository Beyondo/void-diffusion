import re
from io import BytesIO
from typing import Optional

import requests
import torch
from transformers import (
    AutoFeatureExtractor,
    BertTokenizerFast,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    PriorTransformer,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableUnCLIPImg2ImgPipeline,
    StableUnCLIPPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.paint_by_example import PaintByExamplePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_pipeline_from_original_stable_diffusion_ckpt(
    checkpoint_path: str,
    original_config_file: str = None,
    num_in_channels: int = None,
    scheduler_type: str = "pndm",
    pipeline_type: str = None,
    image_size: int = None,
    prediction_type: str = None,
    extract_ema: bool = True,
    upcast_attn: bool = False,
    vae: AutoencoderKL = None,
    vae_path: str = None,
    precision: torch.dtype = torch.float32,
    return_generator_pipeline: bool = False
) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    :param checkpoint_path: Path to `.ckpt` file.
    :param original_config_file: Path to `.yaml` config file corresponding to the original architecture.
      If `None`, will be automatically inferred by looking for a key that only exists in SD2.0 models.
    :param image_size: The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
      Base. Use 768 for Stable Diffusion v2.
    :param prediction_type: The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion
     v1.X and Stable Diffusion v2 Base. Use `'v-prediction'` for Stable Diffusion v2.
    :param num_in_channels: The number of input channels. If `None` number of input channels will be automatically
    inferred.
    :param scheduler_type: Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler",
     "euler-ancestral", "dpm", "ddim"]`. :param model_type: The pipeline type. `None` to automatically infer, or one of
     `["FrozenOpenCLIPEmbedder", "FrozenCLIPEmbedder", "PaintByExample"]`. :param extract_ema: Only relevant for
     checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights
     or not. Defaults to `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher
     quality images for inference. Non-EMA weights are usually better to continue fine-tuning.
    :param precision: precision to use - torch.float16, torch.float32 or torch.autocast
    :param upcast_attention: Whether the attention computation should always be upcasted. This is necessary when
    running stable diffusion 2.1.
    :param vae: A diffusers VAE to load into the pipeline.
    :param vae_path: Path to a checkpoint VAE that will be converted into diffusers and loaded into the pipeline.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verbosity = dlogging.get_verbosity()
        dlogging.set_verbosity_error()

        if checkpoint_path.endswith('.ckpt'):
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = load_file(checkpoint_path)

        cache_dir = "/content/cache_dir"
        pipeline_class = ( StableDiffusionPipeline )

        # Sometimes models don't have the global_step item
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            print("   | global_step key not found in model")
            global_step = None

        # sometimes there is a state_dict key and sometimes not
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        upcast_attention = False
        if original_config_file is None:
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"

            # model_type = "v1"
            config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"

            if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
                # model_type = "v2"
                config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"

                if global_step == 110000:
                    # v2.1 needs to upcast attention
                    upcast_attention = True

            original_config_file = BytesIO(requests.get(config_url).content)

        original_config = OmegaConf.load(original_config_file)

        if num_in_channels is not None:
            original_config["model"]["params"]["unet_config"]["params"][
                "in_channels"
            ] = num_in_channels

        if (
            "parameterization" in original_config["model"]["params"]
            and original_config["model"]["params"]["parameterization"] == "v"
        ):
            if prediction_type is None:
                # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
                # as it relies on a brittle global step parameter here
                prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
            if image_size is None:
                # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
                # as it relies on a brittle global step parameter here
                image_size = 512 if global_step == 875000 else 768
        else:
            if prediction_type is None:
                prediction_type = "epsilon"
            if image_size is None:
                image_size = 512

        num_train_timesteps = original_config.model.params.timesteps
        beta_start = original_config.model.params.linear_start
        beta_end = original_config.model.params.linear_end

        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
        # make sure scheduler works correctly with DDIM
        scheduler.register_to_config(clip_sample=False)

        if scheduler_type == "pndm":
            config = dict(scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(config)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        elif scheduler_type == "ddim":
            scheduler = scheduler
        else:
            raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

        # Convert the UNet2DConditionModel model.
        unet_config = create_unet_diffusers_config(
            original_config, image_size=image_size
        )
        unet_config["upcast_attention"] = upcast_attention
        unet = UNet2DConditionModel(**unet_config)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
        )

        unet.load_state_dict(converted_unet_checkpoint)

        # If a replacement VAE path was specified, we'll incorporate that into
        # the checkpoint model and then convert it
        if vae_path:
            print(f"   | Converting VAE {vae_path}")
            replace_checkpoint_vae(checkpoint,vae_path)
        # otherwise we use the original VAE, provided that
        # an externally loaded diffusers VAE was not passed
        elif not vae:
            print("   | Using checkpoint model's original VAE")

        if vae:
            print("   | Using replacement diffusers VAE")
        else:  # convert the original or replacement VAE
            vae_config = create_vae_diffusers_config(
                original_config, image_size=image_size
            )
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                checkpoint, vae_config
            )

            vae = AutoencoderKL(**vae_config)
            vae.load_state_dict(converted_vae_checkpoint)

        # Convert the text model.
        model_type = pipeline_type
        if model_type is None:
            model_type = original_config.model.params.cond_stage_config.target.split(
                "."
            )[-1]

        if model_type == "FrozenOpenCLIPEmbedder":
            text_model = convert_open_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-2",
                subfolder="tokenizer",
                cache_dir=cache_dir,
            )
            pipe = pipeline_class(
                vae=vae.to(precision),
                text_encoder=text_model.to(precision),
                tokenizer=tokenizer,
                unet=unet.to(precision),
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        elif model_type == "PaintByExample":
            vision_model = convert_paint_by_example_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=cache_dir
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", cache_dir=cache_dir
            )
            pipe = PaintByExamplePipeline(
                vae=vae,
                image_encoder=vision_model,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
            )
        elif model_type in ["FrozenCLIPEmbedder", "WeightedFrozenCLIPEmbedder"]:
            text_model = convert_ldm_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir=cache_dir
            )
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                cache_dir="/content/cache_dir",
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", cache_dir=cache_dir
            )
            pipe = pipeline_class(
                vae=vae.to(precision),
                text_encoder=text_model.to(precision),
                tokenizer=tokenizer,
                unet=unet.to(precision),
                scheduler=scheduler,
                safety_checker=None if return_generator_pipeline else safety_checker.to(precision),
                feature_extractor=feature_extractor,
            )
        else:
            text_config = create_ldm_bert_config(original_config)
            text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
            tokenizer = BertTokenizerFast.from_pretrained(
                "bert-base-uncased", cache_dir=cache_dir
            )
            pipe = LDMTextToImagePipeline(
                vqvae=vae,
                bert=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
            )
    dlogging.set_verbosity(verbosity)

    return pipe

import torch
def from_pretrained(checkpoint_path):
    pipe = None
    try:
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(
            checkpoint_path,
            precision=torch.float16
        )
        pipe.to('cuda')
    except Exception as e:
        print("Failed to load checkpoint %s: %s" % (checkpoint_path, e))
    return pipe