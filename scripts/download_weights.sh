#!/usr/bin/env bash
# Downloads the Depth Anything V2 (Small) checkpoint into ./checkpoints
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints

f=checkpoints/depth_anything_v2_vits.pth
[ -f $f ] || wget -O $f https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# The SDXL inpainting model (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
# is downloaded automatically by diffusers on the first run.
