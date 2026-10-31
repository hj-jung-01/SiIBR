import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sr.pft import PFTSuperResolver, DEFAULT_PFT_ROOT

# Super-resolves an image or every image in a folder with PFT.
# Output names follow PFT-SR/inference.py: <name>_PFT_<task>_SRx<scale>.png

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, required=True, help="input image or directory")
    parser.add_argument("-o", "--out_path", type=str, default="results/sr", help="output directory")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="SR scale factor")
    parser.add_argument("--task", type=str, default="classical", choices=['classical', 'lightweight'])
    parser.add_argument("--tile", type=int, default=0, help="tile size for tiled inference (0: whole image)")
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--pft_root", type=str, default=DEFAULT_PFT_ROOT, help="path to the PFT-SR repository")
    return parser.parse_args()


def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr = PFTSuperResolver(args.scale, args.task, args.pft_root, device, args.tile, args.tile_overlap)
    os.makedirs(args.out_path, exist_ok=True)

    if os.path.isdir(args.in_path):
        files = sorted(os.path.join(args.in_path, f) for f in os.listdir(args.in_path)
                       if f.lower().endswith(EXTS))
    else:
        files = [args.in_path]

    for f in files:
        img = transforms.ToTensor()(Image.open(f).convert('RGB')).unsqueeze(0)
        out = sr(img)[0].cpu()
        name = os.path.splitext(os.path.basename(f))[0]
        out_file = os.path.join(args.out_path, f'{name}_PFT_{args.task}_SRx{args.scale}.png')
        transforms.ToPILImage()(out).save(out_file)
        print(f'{f} {tuple(img.shape[-2:])} -> {out_file} {tuple(out.shape[-2:])}')


if __name__ == "__main__":
    main()
