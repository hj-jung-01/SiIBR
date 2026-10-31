import os
import random
import argparse
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data.dataloader_4d import LF4DDataset
from model.siibr_net import SiIBR, relative_disparity
from model.depth_anything_v2.dpt import DepthAnythingV2, MODEL_CONFIGS
from loss.loss import EntropyLoss, StructureTensorLoss

# SiIBR training
#   L = 100 L_c + 10 L_f + 1 L_w + 0.1 L_epi
#   L_c: color MSE, L_f: MSE between FFTs, L_w: entropy of the blending weights,
#   L_epi: L1 between EPI structure tensors.
# Each iteration renders the EPIs of `num_epi` random rows over the center-row views and a
# random `patch` x `patch` region of one random view of the 4D grid, from the bicubic
# up-sampled low-resolution center view.


def str2bool(v):
    return v if isinstance(v, bool) else v.lower() in ('yes', 'true', 't', '1')


def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--lfinpath', type=str, default='./LFdata/')
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--crop_height', type=int, default=256, help='high-resolution crop')
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--angular_resolution', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--scale', type=int, default=2, help='super-resolution factor')
    # model
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--prompt_dim', type=int, default=64)
    parser.add_argument('--num_related', type=int, default=5)
    parser.add_argument('--baseline', type=float, default=0.01)
    parser.add_argument('--depth_encoder', type=str, default='vits')
    parser.add_argument('--depth_ckpt_dir', type=str, default='checkpoints')
    # training
    parser.add_argument('--iterations', type=int, default=630000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epi', type=int, default=2, help='EPI rows rendered per iteration')
    parser.add_argument('--patch', type=int, default=24, help='size of the random 4D view patch')
    parser.add_argument('--chunk', type=int, default=4096, help='rays per transformer call')
    parser.add_argument('--w_c', type=float, default=100.)
    parser.add_argument('--w_f', type=float, default=10.)
    parser.add_argument('--w_w', type=float, default=1.)
    parser.add_argument('--w_epi', type=float, default=0.1)
    # logging
    parser.add_argument('--save_path', type=str, default='./siibr_checkpoint')
    parser.add_argument('--tb_log_path', type=str, default='./tb_logs_siibr')
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--load_model_name', type=str, default=None, help='resume from a checkpoint')
    parser.add_argument('--amp', type=str2bool, default=False, help='mixed precision')
    return parser.parse_args()


def make_lr(hr, scale):
    # low-resolution input (bicubic down-sampling) and its bicubic up-sampling to the target grid
    H, W = hr.shape[-2:]
    lr = F.interpolate(hr, (H//scale, W//scale), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    up = F.interpolate(lr, (H, W), mode='bicubic', align_corners=False).clamp(0, 1)
    return lr, up


def fft_loss(pred, gt):
    return (torch.fft.fft2(pred, norm='ortho') - torch.fft.fft2(gt, norm='ortho')).abs().pow(2).mean()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.tb_log_path)

    dataset = LF4DDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print('train scenes :', len(dataset))

    model = SiIBR(args.dim, args.depth, args.heads, args.prompt_dim, args.num_related, args.baseline).to(device)
    print(f'SiIBR parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    depth_model = DepthAnythingV2(**MODEL_CONFIGS[args.depth_encoder])
    depth_model.load_state_dict(torch.load(os.path.join(args.depth_ckpt_dir, f'depth_anything_v2_{args.depth_encoder}.pth'),
                                           map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    for p in depth_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    entropy_loss = EntropyLoss()
    st_loss = StructureTensorLoss()

    it = 0
    if args.load_model_name is not None:
        ckpt = torch.load(args.load_model_name, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        it = ckpt['iteration']
        print(f'resume from iteration {it}')

    AN = args.angular_resolution
    c = AN // 2
    model.train()
    while it < args.iterations:
        for center, row, view, uv in loader:
            center, row, view = center.to(device), row.to(device), view.to(device)
            B, _, h, w = center.shape

            lr, src_hr = make_lr(center, args.scale)
            rel = relative_disparity(depth_model, src_hr)

            with torch.autocast('cuda', enabled=args.amp):
                src = model.make_source(src_hr, lr, rel)

                # EPIs over the center row
                epi_rows = sorted(random.sample(range(h), args.num_epi))
                epi_pred, weights = [], []
                for u in range(AN):
                    rgb, _, _, _, wgt = model.render([src], u - c, 0., rows=epi_rows, chunk=args.chunk)
                    epi_pred.append(rgb)
                    weights.append(wgt)
                epi_pred = torch.stack(epi_pred, 2)                       # [B,3,AN,num_epi,w]
                epi_gt = row[:, :, :, epi_rows]                           # [B,3,AN,num_epi,w]

                # random patch of a random 4D view (each sample has its own view)
                y0, x0 = random.randint(0, h - args.patch), random.randint(0, w - args.patch)
                rows, cols = list(range(y0, y0 + args.patch)), list(range(x0, x0 + args.patch))
                patch_pred = []
                for b in range(B):
                    src_b = {k: (t[b:b+1] if torch.is_tensor(t) else t) for k, t in src.items()}
                    rgb, _, _, _, wgt = model.render([src_b], uv[b, 0].item(), uv[b, 1].item(),
                                                     rows=rows, cols=cols, chunk=args.chunk)
                    patch_pred.append(rgb)
                    weights.append(wgt)
                patch_pred = torch.cat(patch_pred, 0)
                patch_gt = view[:, :, y0:y0 + args.patch, x0:x0 + args.patch]

            epi_pred, patch_pred = epi_pred.float(), patch_pred.float()
            epi_img = epi_pred.permute(0, 3, 1, 2, 4).reshape(-1, 3, AN, w)
            epi_img_gt = epi_gt.permute(0, 3, 1, 2, 4).reshape(-1, 3, AN, w)

            losses = {
                'color': args.w_c * (F.mse_loss(epi_pred, epi_gt) + F.mse_loss(patch_pred, patch_gt)),
                'fft': args.w_f * fft_loss(patch_pred, patch_gt),
                'entropy': args.w_w * entropy_loss(torch.cat(weights, 0).float()[..., None]),
                'epi_st': args.w_epi * st_loss(epi_img_gt, epi_img),
            }
            loss = sum(losses.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            it += 1

            writer.add_scalars('loss', {'total': loss.item(), **{k: v.item() for k, v in losses.items()}}, it)
            if it % args.print_every == 1:
                print(f'iter {it} | loss {loss.item():.5f} | ' + ' '.join(f'{k} {v.item():.5f}' for k, v in losses.items()))
                writer.add_image('01_input_lr_up', make_grid(src_hr[:2].cpu()), it)
                writer.add_image('02_epi_gt', make_grid(epi_img_gt[:4].cpu(), nrow=1), it)
                writer.add_image('03_epi_pred', make_grid(epi_img[:4].detach().clamp(0, 1).cpu(), nrow=1), it)
                writer.add_image('04_patch', make_grid(torch.cat([patch_gt[:1], patch_pred[:1].detach().clamp(0, 1)]).cpu()), it)
            if it % args.save_every == 0 or it == args.iterations:
                timestamp = datetime.datetime.now().strftime('%m-%d-%H_%M')
                torch.save({'iteration': it,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'args': vars(args)},
                           os.path.join(args.save_path, f'siibr_x{args.scale}-{timestamp}-iter{it}.pth'))
            if it >= args.iterations:
                break
    writer.close()


if __name__ == '__main__':
    main()
