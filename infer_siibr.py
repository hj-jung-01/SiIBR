import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import skimage.io as io

from utils import normalize
from model.siibr_net import SiIBR, relative_disparity
from model.depth_anything_v2.dpt import DepthAnythingV2, MODEL_CONFIGS

# High-resolution 4D light field from a single low-resolution image with SiIBR.
#
#   1. the input is bicubic up-sampled by --scale and its disparity is D = alpha * F(I) + beta
#   2. target views are rendered on the dense grid, farthest from the center first
#   3. dis-occluded pixels (high weight entropy) are inpainted and added to the source rays
#      (up to --max_sources source views)
#   4. the center view rendered at the input pose is the super-resolved input


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='low-resolution input image')
    parser.add_argument('--ckpt', type=str, required=True, help='SiIBR checkpoint from train_siibr.py')
    parser.add_argument('--out_dir', type=str, default='results_siibr')
    parser.add_argument('--max_side', type=int, default=320, help='longer side of the low-resolution input (0: keep)')
    parser.add_argument('--angular_resolution', type=int, default=11)
    parser.add_argument('--depth_ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--inpainter', type=str, default='sdxl', choices=['sdxl', 'none'])
    parser.add_argument('--prompt', type=str, default='fill as background', help='prompt for the SDXL inpainter')
    parser.add_argument('--entropy_thresh', type=float, default=1.5, help='occlusion threshold k on the weight entropy (bits)')
    parser.add_argument('--max_sources', type=int, default=5, help='max. number of source views (input + inpainted views)')
    parser.add_argument('--row_chunk', type=int, default=32, help='rows rendered per call')
    parser.add_argument('--chunk', type=int, default=8192, help='rays per transformer call')
    parser.add_argument('--save_npy', action='store_true', help='also save the light field as [U,V,H,W,3] uint8')
    return parser.parse_args()


def to_uint8(x):
    return (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()*255).round().astype('uint8')


def main():
    args = get_args()
    os.makedirs(os.path.join(args.out_dir, 'views'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    targs = ckpt['args']
    scale = targs['scale']
    model = SiIBR(targs['dim'], targs['depth'], targs['heads'], targs['prompt_dim'],
                  targs['num_related'], targs['baseline']).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    depth_model = DepthAnythingV2(**MODEL_CONFIGS[targs['depth_encoder']])
    depth_model.load_state_dict(torch.load(os.path.join(args.depth_ckpt_dir, f"depth_anything_v2_{targs['depth_encoder']}.pth"),
                                           map_location='cpu'))
    depth_model = depth_model.to(device).eval()

    inpainter = None
    if args.inpainter == 'sdxl':
        from model.sdxl_inpainter import Inpainter
        inpainter = Inpainter(device=device, prompt=args.prompt)

    # low-resolution input
    lr = io.imread(args.image)[..., :3].astype('float32') / 255.
    lr = torch.from_numpy(lr).permute(2, 0, 1)[None].to(device)
    if args.max_side > 0 and max(lr.shape[-2:]) > args.max_side:
        r = args.max_side / max(lr.shape[-2:])
        lr = F.interpolate(lr, (round(lr.shape[-2]*r), round(lr.shape[-1]*r)), mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1)
    h, w = lr.shape[-2:]
    H, W = h*scale, w*scale
    print(f'input {w}x{h} -> light field views {W}x{H} (x{scale})')

    AN = args.angular_resolution
    c = AN // 2
    with torch.no_grad():
        src_hr = F.interpolate(lr, (H, W), mode='bicubic', align_corners=False).clamp(0, 1)
        io.imsave(os.path.join(args.out_dir, 'input_bicubic.png'), to_uint8(src_hr), check_contrast=False)
        rel = relative_disparity(depth_model, src_hr)
        sources = [model.make_source(src_hr, lr, rel)]
        io.imsave(os.path.join(args.out_dir, 'input_disparity.png'),
                  normalize(sources[0]['disp_x'][0, 0].cpu().numpy()), check_contrast=False)

        # farthest views first, the input pose last
        grid = [(u, v) for u in range(-c, AN - c) for v in range(-c, AN - c)]
        grid.sort(key=lambda t: -(t[0]**2 + t[1]**2))
        lf = np.zeros((AN, AN, H, W, 3), dtype=np.uint8)
        for n, (u, v) in enumerate(grid):
            rgb, disp, ent, dist = [], [], [], []
            for y in range(0, H, args.row_chunk):
                out = model.render(sources, float(u), float(v), rows=list(range(y, min(y + args.row_chunk, H))),
                                   chunk=args.chunk)
                rgb.append(out[0]); disp.append(out[1]); ent.append(out[2]); dist.append(out[3])
            rgb, disp, ent, dist = [torch.cat(t, 2) for t in (rgb, disp, ent, dist)]
            valid = ((ent < args.entropy_thresh) & (dist < 1)).float()

            n_hole = (1 - valid).sum().item()
            if inpainter is not None and len(sources) < args.max_sources and 0 < n_hole and valid.sum() > 0:
                rgb = inpainter(rgb, valid)
                rel_new = relative_disparity(depth_model, rgb)
                disp_new = valid*disp + (1 - valid)*model.disp_head(rgb, rel_new)
                lr_new = F.interpolate(rgb, (h, w), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                # only the generated pixels become new source rays
                sources.append(model.make_source(rgb, lr_new, rel_new, u, v, mask=1 - valid, disp=disp_new))
                print(f'  view ({u:+d},{v:+d}): inpainted {int(n_hole)} px, {len(sources)} source views')

            lf[u + c, v + c] = to_uint8(rgb)
            io.imsave(os.path.join(args.out_dir, 'views', f'view_{u+c:02d}_{v+c:02d}.png'), lf[u + c, v + c], check_contrast=False)
            print(f'[{n+1}/{len(grid)}] view ({u:+d},{v:+d}) done')

    io.imsave(os.path.join(args.out_dir, 'sr_input_pose.png'), lf[c, c], check_contrast=False)
    io.imsave(os.path.join(args.out_dir, 'epi_center_row.png'), np.repeat(lf[:, c, H//2], 8, axis=0), check_contrast=False)
    io.imsave(os.path.join(args.out_dir, 'epi_center_col.png'),
              np.repeat(lf[c, :, :, W//2].transpose(1, 0, 2), 8, axis=1), check_contrast=False)
    if args.save_npy:
        np.save(os.path.join(args.out_dir, 'lightfield.npy'), lf)
    print('Saved to', args.out_dir)


if __name__ == '__main__':
    main()
