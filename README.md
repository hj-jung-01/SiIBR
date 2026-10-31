# Reconstruction and Restoration of 4D Light Fields via Super-resolved Inverse Image-Based Rendering

Hyunjun Jung, Hae-Gon Jeon

International Journal of Computer Vision (IJCV), 2026 [[Paper]](https://link.springer.com/article/10.1007/s11263-026-03111-8)

> **Note.** This code has **not been verified to reproduce the results reported in the paper**.
> Training and inference run end to end, but pretrained SiIBR weights are not provided and the
> numbers in the paper were not re-measured with this release.

SiIBR reconstructs a high-resolution 4D light field from a single low-resolution image.
The input is bicubic up-sampled, and a ray transformer renders every target ray on the dense grid as a
weighted sum of source rays found along the horizontal and vertical EPIs (inverse image-based rendering).
Pixel-aligned SR prompt tokens (image gradients plus low- and high-frequency prompts) modulate the source
rays, so spatial super-resolution and EPI slopes are learned together. Dis-occluded regions are inpainted
and added back to the source rays for the following views.

## Repository layout

```
train_siibr.py             training
infer_siibr.py             single low-resolution image -> high-resolution 4D light field
model/siibr_net.py         SiIBR (4D candidate rays, ray transformer, disparity scale/shift)
model/sr_prompt.py         pixel-aligned SR prompt tokens
model/sdxl_inpainter       inpainter for dis-occluded regions (SDXL-inpainting)
model/depth_anything_v2    Depth Anything V2 (third-party, Apache-2.0)
data/dataloader_4d.py      4D light field loader
loss/loss.py               entropy and EPI structure tensor losses
sr/, scripts/              PFT super-resolution tools for the cascaded SR-then-NVS baselines
```

## Installation

Tested with Python 3.11 and PyTorch 2.5.0 (CUDA 12.1) on a V100 (32GB).

```bash
pip install -r requirements.txt
bash scripts/download_weights.sh      # Depth Anything V2 (Small) -> checkpoints/
```

The SDXL inpainting model (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, ~7GB) is downloaded by
`diffusers` on the first run.

## Dataset

The model is trained on the synthetic light field dataset of Heber et al. [1] (https://www.etaargus.com/),
900 scenes with 11x11 sub-aperture images. The loader expects

```
LFdata/
  <scene_name>/                          # folder names contain "scene"
    scene000.png ... scene120.png        # 11x11 views; view (u, v) is scene{u*11+v}
```

The first 100 scenes (sorted by name) are held out for testing.

## Training

```bash
python train_siibr.py --lfinpath /path/to/LFdata --scale 2 --depth_ckpt_dir checkpoints
```

* The high-resolution crop (`--crop_height/--crop_width`, default 256) is bicubic down-sampled by
  `--scale` to make the low-resolution input. One model is trained per scale.
* Each iteration renders the EPIs of `--num_epi` random rows over the center-row views and a random
  `--patch` x `--patch` region of a random view of the 4D grid.
* Loss: `100 * L_color + 10 * L_fft + 1 * L_entropy + 0.1 * L_epi-structure-tensor`
  (`--w_c`, `--w_f`, `--w_w`, `--w_epi`). Adam (0.9, 0.99), learning rate 1e-4, batch size 1.
* Transformer: 6 layers, 12 heads, 768 dimensions (`--depth`, `--heads`, `--dim`); 5 source rays per
  target ray (`--num_related`). With the defaults training takes about 12GB of GPU memory.

## Inference

```bash
python infer_siibr.py --image path/to/low_res.png --ckpt siibr_checkpoint/<checkpoint>.pth --out_dir results_siibr
```

* `--max_side` (default 320): longer side of the low-resolution input; views are `scale` times larger.
* `--angular_resolution` (default 11): number of views per angular axis.
* Views are rendered farthest from the center first. Pixels whose blending-weight entropy is above
  `--entropy_thresh` are inpainted (`--inpainter sdxl|none`), and the generated pixels are added to the
  source rays (up to `--max_sources` source views).
* Lower `--row_chunk` / `--chunk` to reduce memory.

Outputs: `views/view_UU_VV.png` (UU, VV = 00 ... 10), `sr_input_pose.png` (the center view, i.e. the
super-resolved input), `input_bicubic.png`, `input_disparity.png`, EPIs of the center row / column, and
`lightfield.npy` (`[U, V, H, W, 3]`, with `--save_npy`).

## Super-resolution baselines

The cascaded baselines first super-resolve the low-resolution input and then synthesize views from it.
PFT [3] is installed with

```bash
bash scripts/setup_pft_sr.sh          # -> third_party/PFT-SR  (or set PFT_SR_ROOT=/path/to/PFT-SR)
```

and its official weights go to `third_party/PFT-SR/experiments/pretrained_models/`.

```bash
# super-resolve an image or a folder with PFT
python sr/super_resolve.py -i path/to/images/ -o results/sr --scale 4

# down-scale images by 1/n (n = 2, 3, 4, 8), restore them with PFT (x2, x3, x4, x4)
# and report PSNR / SSIM against the originals
bash scripts/sr_experiment.sh path/to/images results/sr_experiment
```

## Acknowledgements and third-party code

* [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) [2]: code in `model/depth_anything_v2` (Apache-2.0).
* [PFT-SR](https://github.com/LabShuHangGU/PFT-SR) [3] (Apache-2.0), cloned by `scripts/setup_pft_sr.sh`.
* The SR prompt modules follow the low/high-frequency prompt modules of PromptRestorer.
* [SDXL inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) via `diffusers`.

## License

MIT License (see `LICENSE`). Third-party code keeps its own license.

## References

[1] S. Heber and T. Pock. Convolutional networks for shape from light field. CVPR, 2016.  
[2] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. Depth Anything V2. NeurIPS, 2024.  
[3] W. Long, X. Zhou, L. Zhang, and S. Gu. Progressive Focused Transformer for Single Image Super-Resolution. CVPR, 2025.
