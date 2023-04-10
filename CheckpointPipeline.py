from diffusers import StableDiffusionPipeline
import torch, os, importlib
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt

def convert_checkpoint(checkpoint_path, original_config_file=None, num_in_channels=None, 
                       scheduler_type='pndm', pipeline_type=None, image_size=None, prediction_type=None, 
                       extract_ema=False, upcast_attention=False, from_safetensors=False, to_safetensors=False, 
                       dump_path=None, device=None, stable_unclip=None, stable_unclip_prior=None, clip_stats_path=None, 
                       controlnet=None, is_img2img=False):
    
    args = argparse.Namespace(checkpoint_path=checkpoint_path,
                              original_config_file=original_config_file,
                              num_in_channels=num_in_channels,
                              scheduler_type=scheduler_type,
                              pipeline_type=pipeline_type,
                              image_size=image_size,
                              prediction_type=prediction_type,
                              extract_ema=extract_ema,
                              upcast_attention=upcast_attention,
                              from_safetensors=from_safetensors,
                              to_safetensors=to_safetensors,
                              dump_path=dump_path,
                              device=device,
                              stable_unclip=stable_unclip,
                              stable_unclip_prior=stable_unclip_prior,
                              clip_stats_path=clip_stats_path,
                              controlnet=controlnet)
    
    if args.from_safetensors:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device), pickle_module=dill)
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))

    if args.original_config_file is None:
        # infer original config file from checkpoint
        assert checkpoint['hyper_parameters']['config'] == 'stable_diffusion'
        if checkpoint['hyper_parameters']['model_config']['dequantize'] == 'bits(8)':
            args.original_config_file = 'configs/stable-diffusion/v1-inference.yaml'
        else:
            raise ValueError(f'Cannot automatically infer original config file from checkpoint: {args.checkpoint_path}')

    config = load_pipeline_from_original_stable_diffusion_ckpt(args.original_config_file, load_safety_checker=False, is_img2img=is_img2img)

    if args.pipeline_type is None:
        args.pipeline_type = config.pipeline_type

    if args.image_size is None:
        args.image_size = config.image_size

    if args.prediction_type is None:
        args.prediction_type = config.prediction_type

    if args.num_in_channels is None:
        args.num_in_channels = config.num_in_channels

    pipeline_cls = config.get_pipeline_cls(args.pipeline_type)

    pipeline = pipeline_cls.from_pretrained(
        checkpoint,
        num_in_channels=args.num_in_channels,
        scheduler_type=args.scheduler_type,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        upcast_attention=args.upcast_attention,
        extract_ema=args.extract_ema,
        device=args.device,
        stable_unclip=args.stable_unclip,
        stable_unclip_prior=args.stable_unclip_prior,
        clip_stats_path=args.clip_stats_path,
        controlnet=args.controlnet,
    )
    
    if args.to_safetensors:
        pipeline.save(args.dump_path, with_safetensors=True)
    else:
        pipeline.save(args.dump_path)
    
    return pipeline

def from_pretrained(checkpoint_path, is_img2img=False):
    torch.set_default_dtype(torch.float16)
    pipe = None
    try:
        pipe = convert_checkpoint(ModelName, is_img2img)
    except Exception as e:
        print("Failed to load model %s: %s" % (checkpoint_path, e))
    pipe.to("cuda:0")
    return pipe