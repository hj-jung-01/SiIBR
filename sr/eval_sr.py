import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# PSNR / SSIM between super-resolved images and the original (full-resolution) images.
# Images are matched by name: <name>.png  <->  <name>_PFT_<task>_SRx<scale>.png
# When the sizes differ by a few pixels (floor in downscaling), both are cropped
# to the common top-left region. When the SR output is smaller than the original
# (e.g. 1/8 input restored with x4), the original is downscaled to the SR size.

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def load(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float64) / 255.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='original images')
    parser.add_argument('--sr_dir', type=str, required=True, help='super-resolved images')
    args = parser.parse_args()

    gts = {os.path.splitext(f)[0]: f for f in os.listdir(args.gt_dir) if f.lower().endswith(EXTS)}
    rows = []
    for f in sorted(os.listdir(args.sr_dir)):
        if not f.lower().endswith(EXTS):
            continue
        name = os.path.splitext(f)[0].split('_PFT_')[0]
        if name not in gts:
            continue
        sr = load(os.path.join(args.sr_dir, f))
        gt = load(os.path.join(args.gt_dir, gts[name]))
        if gt.shape[0] - sr.shape[0] > 8 or gt.shape[1] - sr.shape[1] > 8:
            gt = load(os.path.join(args.gt_dir, gts[name]), (sr.shape[1], sr.shape[0]))
        h, w = min(gt.shape[0], sr.shape[0]), min(gt.shape[1], sr.shape[1])
        gt, sr = gt[:h, :w], sr[:h, :w]
        psnr = peak_signal_noise_ratio(gt, sr, data_range=1.)
        ssim = structural_similarity(gt, sr, data_range=1., channel_axis=-1)
        rows.append((name, psnr, ssim))
        print(f'{name:30s} PSNR {psnr:6.2f}  SSIM {ssim:.4f}')
    if rows:
        print(f'{"mean":30s} PSNR {np.mean([r[1] for r in rows]):6.2f}  SSIM {np.mean([r[2] for r in rows]):.4f}')


if __name__ == '__main__':
    main()
