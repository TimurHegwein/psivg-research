#!/usr/bin/env python3
"""Standalone CogVideoX-5B-I2V (+ Go-with-the-Flow LoRA) video generator.

This is the *physics-free* generator: given a single input image and a text
prompt it produces a 49-frame video with NO physical-simulator guidance and NO
TTCO optimisation. That makes it useful for two things:

  1. TEMPLATE / preview videos for brand-new examples (paper, robotics, ...)
     before we invest in the full 4-step pipeline.
  2. The NEGATIVE baseline for the paper: "what plain video generation does
     without our physics-in-the-loop guidance" (typically physically wrong).

It reuses the exact LoRA-loading path from
psivg/video_generation/cogvideox_image_to_video_lora.py (LoraConfig r=2048,
load Go-with-the-Flow weights into the transformer) so the backbone matches the
real pipeline. Inference only -> fits comfortably in <40 GB, runs on the H100.

Run from the repo root inside PSIVG_env3:
  conda activate PSIVG_env3
  python research_timur/gen_template_video.py \
      --image path/to/first_frame.png \
      --prompt "a basketball bouncing on a wooden floor" \
      --output research_timur/template_videos/basketball.mp4

Optional --noise <warped_noise.npy> feeds physics-warped noise (the same format
Steps 1-3 produce in <video_id>/noises/) if you want a guided generation; omit
it for the plain baseline.
"""
import os
import argparse
import numpy as np
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from peft import LoraConfig, set_peft_model_state_dict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BASE = os.path.join(REPO, "pretrained_models/CogVideoX-5b-I2V")
DEFAULT_LORA = os.path.join(
    REPO, "pretrained_models/I2V5B_final_i38800_nearest_lora_weights.safetensors")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="first-frame conditioning image")
    p.add_argument("--prompt", required=True, help="text prompt")
    p.add_argument("--output", required=True, help="output .mp4 path")
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--lora", default=DEFAULT_LORA,
                   help="Go-with-the-Flow LoRA weights; pass 'none' to disable")
    p.add_argument("--rank", type=int, default=2048)
    p.add_argument("--lora_alpha", type=int, default=2048)
    p.add_argument("--noise", default=None,
                   help="optional warped-noise .npy (Steps 1-3 format). "
                        "Omit for the plain physics-free baseline.")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--degradation", type=float, default=0.5,
                   help="only used when --noise is given (matches pipeline default)")
    return p.parse_args()


def load_lora(transformer, lora_path, rank, alpha):
    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights=True,
                     target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    transformer.add_adapter(cfg)
    sd = LoraLoaderMixin.lora_state_dict(lora_path)[0]
    sd = {k.replace("transformer.", ""): v for k, v in sd.items()
          if k.startswith("transformer.")}
    sd = convert_unet_state_dict_to_peft(sd)
    set_peft_model_state_dict(transformer, sd, adapter_name="default")
    print("Loaded LoRA:", lora_path)


def main():
    args = parse_args()
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {args.base_model}")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.base_model, torch_dtype=dtype)

    if args.lora and args.lora.lower() != "none":
        load_lora(pipe.transformer, args.lora, args.rank, args.lora_alpha)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe = pipe.to(device)

    # Optional physics-warped noise (else the pipeline samples plain Gaussian).
    latents = None
    if args.noise:
        import rp.git.CommonSource.noise_warp as nw  # type: ignore
        from psivg.video_generation.dataset import get_downtemp_noise  # type: ignore
        n = np.load(args.noise)
        n = torch.tensor(n).permute(0, 3, 1, 2).to(dtype)
        n = get_downtemp_noise(n, "nearest")[None]
        latents = nw.mix_new_noise(n, args.degradation)
        print("Using warped noise:", args.noise)
    else:
        print("Using plain Gaussian noise (physics-free baseline).")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    kwargs = dict(prompt=args.prompt, image=load_image(args.image),
                  height=args.height, width=args.width,
                  num_frames=args.num_frames,
                  num_inference_steps=args.steps,
                  guidance_scale=args.guidance_scale,
                  generator=generator, output_type="np")
    if latents is not None:
        kwargs["latents"] = latents

    with torch.no_grad():
        video = pipe(**kwargs).frames[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    export_to_video(video, args.output, fps=args.fps)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
