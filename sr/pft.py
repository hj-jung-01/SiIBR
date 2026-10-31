import os
import sys
import torch
import torch.nn.functional as F

# PFT-SR (Long et al., CVPR 2025) is used as an off-the-shelf super-resolution model.
# It is not vendored: run `bash scripts/setup_pft_sr.sh` to clone it into
# third_party/PFT-SR (pinned commit) and to build its sparse-matmul CUDA op.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PFT_ROOT = os.environ.get('PFT_SR_ROOT', os.path.join(REPO_ROOT, 'third_party', 'PFT-SR'))

MODEL_PATH = {
    "classical": {
        2: "experiments/pretrained_models/001_PFT_SRx2_scratch.pth",
        3: "experiments/pretrained_models/002_PFT_SRx3_finetune.pth",
        4: "experiments/pretrained_models/003_PFT_SRx4_finetune.pth",
    },
    "lightweight": {
        2: "experiments/pretrained_models/101_PFT_light_SRx2_scratch.pth",
        3: "experiments/pretrained_models/102_PFT_light_SRx3_finetune.pth",
        4: "experiments/pretrained_models/103_PFT_light_SRx4_finetune.pth",
    }
}


def build_pft(scale=4, task='classical', pft_root=DEFAULT_PFT_ROOT, device='cuda'):
    '''Builds PFT exactly as in PFT-SR/inference.py and loads the official weights.'''
    if scale not in (2, 3, 4):
        raise ValueError(f'PFT provides x2/x3/x4 models only (got x{scale})')
    if not os.path.isdir(pft_root):
        raise FileNotFoundError(f'PFT-SR not found at {pft_root}; run `bash scripts/setup_pft_sr.sh` '
                                f'or set PFT_SR_ROOT')
    if pft_root not in sys.path:
        sys.path.insert(0, pft_root)
    from basicsr.archs.pft_arch import PFT

    if task == 'classical':
        model = PFT(upscale=scale,
                    embed_dim=240,
                    depths=[4, 4, 4, 6, 6, 6],
                    num_heads=6,
                    num_topk=[1024, 1024, 1024, 1024,
                              256, 256, 256, 256,
                              128, 128, 128, 128,
                              64, 64, 64, 64, 64, 64,
                              32, 32, 32, 32, 32, 32,
                              16, 16, 16, 16, 16, 16, ],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    use_checkpoint=False,
                    )
    elif task == 'lightweight':
        model = PFT(upscale=scale,
                    embed_dim=52,
                    depths=[2, 4, 6, 6, 6],
                    num_heads=4,
                    num_topk=[1024, 1024,
                              256, 256, 256, 256,
                              128, 128, 128, 128, 128, 128,
                              64, 64, 64, 64, 64, 64,
                              32, 32, 32, 32, 32, 32],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=1,
                    upsampler='pixelshuffledirect',
                    use_checkpoint=False,
                    )
    else:
        raise ValueError(f'unknown PFT task: {task}')

    weight_path = os.path.join(pft_root, MODEL_PATH[task][scale])
    state_dict = torch.load(weight_path, map_location='cpu')['params_ema']
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class PFTSuperResolver():
    '''
    Super-resolves RGB tensors [B,3,H,W] in [0,1] by `scale`.
    `tile` > 0 enables overlapping tiled inference for large inputs (e.g. full LF views).
    '''
    def __init__(self, scale=4, task='classical', pft_root=DEFAULT_PFT_ROOT,
                 device='cuda', tile=0, tile_overlap=32):
        self.scale = scale
        self.device = device
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.model = build_pft(scale, task, pft_root, device)

    @torch.no_grad()
    def __call__(self, img):
        img = img.to(self.device)
        if self.tile <= 0 or (img.shape[-2] <= self.tile and img.shape[-1] <= self.tile):
            return self.model(img).clamp(0., 1.)
        return self._tiled(img).clamp(0., 1.)

    def _tiled(self, img):
        B, C, H, W = img.shape
        s, t = self.scale, self.tile
        stride = t - self.tile_overlap
        hs = list(range(0, max(H-t, 0), stride)) + [max(H-t, 0)]
        ws = list(range(0, max(W-t, 0), stride)) + [max(W-t, 0)]
        out = torch.zeros(B, C, H*s, W*s, device=img.device)
        cnt = torch.zeros(B, 1, H*s, W*s, device=img.device)
        for h in hs:
            for w in ws:
                patch = img[..., h:h+t, w:w+t]
                sr = self.model(patch)
                out[..., h*s:h*s+sr.shape[-2], w*s:w*s+sr.shape[-1]] += sr
                cnt[..., h*s:h*s+sr.shape[-2], w*s:w*s+sr.shape[-1]] += 1
        return out / cnt
