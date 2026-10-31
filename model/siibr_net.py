import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sr_prompt import SRPromptEncoder


# SiIBR: super-resolved inverse image-based rendering (IJCV extension of iIBR).
#
# A low-resolution source image is bicubic-upsampled to the target resolution, and every
# target ray on the dense (sH x sW) grid is rendered as a weighted sum of a few source rays
# found along the horizontal and vertical EPIs (4D iIBR). The ray transformer predicts the
# weights from the target ray, the source rays modulated by pixel-aligned SR prompt tokens
# (Eq. 8), and the horizontal / vertical disparity cues (Eq. 9).


def posenc(x, min_deg=1, max_deg=10, torch_channel=True):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi]. shape [B,C,...]
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      torch_channel: True if the second dim is channel [B,C,...] False if the last dim is channel [B,...,C]

    Returns:
      encoded: torch.Tensor, encoded variables. shape [B,C*(2*(max_deg-min_deg)+1),...]
    """
    if min_deg == max_deg:
        return x
    if torch_channel:
        x = x.transpose(1,-1)
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device, dtype=x.dtype)
    xb = torch.reshape(x[..., None, :] * scales[:, None], list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    if torch_channel:
        return torch.cat([x] + [four_feat], dim=-1).transpose(1,-1)
    else:
        return torch.cat([x] + [four_feat], dim=-1)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def relative_disparity(depth_model, img, size=518):
    # Depth Anything V2 on an RGB tensor in [0,1]; returns min-max normalized relative
    # inverse depth (near = 1) at the input resolution
    H, W = img.shape[-2:]
    x = F.interpolate(img, (size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1,3,1,1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1,3,1,1)
    with torch.no_grad():
        d = depth_model((x - mean) / std)[:, None]
    d = F.interpolate(d, (H, W), mode='bilinear', align_corners=False)
    d_min = d.amin(dim=(1,2,3), keepdim=True)
    d_max = d.amax(dim=(1,2,3), keepdim=True)
    return (d - d_min) / (d_max - d_min + 1e-8)


class DisparityScaleShift(nn.Module):
    '''
    D = alpha * F(I) + beta, alpha / beta from the image and its relative disparity
    (five 5x5 conv layers with 256 channels and an MLP that outputs 2 numbers).
    Initialized to alpha=2, beta=-0.3 (the manual mapping used by iIBR).
    '''
    def __init__(self, ch=256, size=128, init_alpha=2.0, init_beta=-0.3):
        super(DisparityScaleShift, self).__init__()
        self.size = size
        layers = []
        in_ch = 4
        for i in range(5):
            layers += [nn.Conv2d(in_ch, ch, 5, 2 if i < 4 else 1, 2), nn.ReLU(inplace=True)]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Linear(ch, 2)
        nn.init.zeros_(self.mlp.weight)
        with torch.no_grad():
            self.mlp.bias.copy_(torch.tensor([init_alpha, init_beta]))

    def forward(self, img, rel_disp):
        x = torch.cat([img, rel_disp], 1)
        x = F.interpolate(x, (self.size, self.size), mode='bilinear', align_corners=False)
        ab = self.mlp(self.conv(x).mean(dim=(2,3)))  # [B,2]
        alpha, beta = ab[:, 0].view(-1,1,1,1), ab[:, 1].view(-1,1,1,1)
        return alpha * rel_disp + beta


class RayTransformer(nn.Module):
    def __init__(self, in_dim, dim=768, depth=6, heads=12, mlp_ratio=4):
        super(RayTransformer, self).__init__()
        self.proj = nn.Linear(in_dim, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*mlp_ratio,
                                           dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, tokens):
        # tokens [N, K, C] -> blending weights [N, K]
        x = self.blocks(self.proj(tokens))
        return F.softmax(self.head(self.norm(x))[..., 0], dim=-1)


class SiIBR(nn.Module):
    def __init__(self, dim=768, depth=6, heads=12, prompt_dim=64, num_related=5,
                 baseline=0.01, pe_deg=10, disp_norm=32.):
        super(SiIBR, self).__init__()
        self.num_related = num_related
        self.baseline = baseline
        self.pe_deg = pe_deg
        self.disp_norm = disp_norm

        self.prompt = SRPromptEncoder(prompt_dim)
        self.disp_head = DisparityScaleShift()

        ray_ch = 6 * (1 + 2*(pe_deg-1))   # positional encoding of Plücker coordinates
        cue_ch = 4 * (1 + 2*(pe_deg-1))   # D_x, D_y and the projected offset of the source ray
        # phi_k: key projection of [source ray, SR prompt] (Eq. 8)
        self.phi_k = nn.Sequential(nn.Linear(ray_ch + prompt_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.transformer = RayTransformer(ray_ch + dim + cue_ch, dim, depth, heads)

    # ------------------------------------------------------------------ rays
    def plucker(self, xs, ys, cam_u, cam_v, H, W):
        # xs, ys: pixel coordinates (any shape), cam_u/cam_v: camera position in view units
        f = 0.75 * W  # same pinhole as iIBR (focal 480 px at 640 px width)
        d = torch.stack([(xs - W/2) / f, (ys - H/2) / f, torch.ones_like(xs)], -1)
        d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
        o = torch.stack([cam_u*self.baseline, cam_v*self.baseline, torch.zeros_like(cam_u)], -1)
        o = o.expand_as(d)
        return posenc(torch.cat([d, torch.linalg.cross(o, d, dim=-1)], -1), 1, self.pe_deg, torch_channel=False)

    # ------------------------------------------------------------------ sources
    def make_source(self, img_hr, img_lr, rel_disp, cam_u=0., cam_v=0., mask=None, disp=None):
        '''
        img_hr: bicubic up-sampled image [B,3,sH,sW], img_lr: low-resolution image (for the prompt)
        rel_disp: relative disparity [B,1,sH,sW] from Depth Anything V2
        mask: 1 for pixels that can be used as source rays
        '''
        if disp is None:
            disp = self.disp_head(img_hr, rel_disp)
        if mask is None:
            mask = torch.ones_like(disp)
        return {'img': img_hr, 'disp_x': disp, 'disp_y': disp, 'mask': mask,
                'prompt': self.prompt(img_hr, img_lr), 'u': float(cam_u), 'v': float(cam_v)}

    # ------------------------------------------------------------------ candidate search
    @staticmethod
    def _nearest_1d(pos, valid, targets, K, chunk=16):
        '''
        pos: projected positions of the source pixels along one line [B,L,S]
        valid: [B,L,S], targets: target coordinates along the line [T]
        returns indices [B,L,T,K] of the K source pixels nearest to each target
        '''
        out = []
        for i in range(0, pos.shape[1], chunk):
            p = pos[:, i:i+chunk]
            m = (p[..., None, :] - targets[None, None, :, None]).abs()          # [B,l,T,S]
            m = torch.where(valid[:, i:i+chunk, None, :] > 0, m, torch.full_like(m, float('inf')))
            out.append(torch.topk(m, K, dim=-1, largest=False).indices)
        return torch.cat(out, 1)

    def find_candidates(self, src, u, v, rows, cols):
        '''
        4D candidate source rays for the target view (u, v) on the target pixels rows x cols.
        Horizontal candidates come from the horizontal EPI (same row), vertical candidates
        from the vertical EPI (same column); their K x K combinations are ranked by the
        2D distance between the re-projected source ray and the target ray.
        returns flat source indices [B,R,C,K] and offsets (dx, dy) [B,R,C,K,2]
        '''
        K = self.num_related
        img = src['img']
        B, _, H, W = img.shape
        du, dv = u - src['u'], v - src['v']
        dx, dy, valid = src['disp_x'][:, 0], src['disp_y'][:, 0], src['mask'][:, 0]
        xs = torch.arange(W, device=img.device, dtype=img.dtype)
        ys = torch.arange(H, device=img.device, dtype=img.dtype)
        rows_t = torch.as_tensor(rows, device=img.device)
        cols_t = torch.as_tensor(cols, device=img.device)

        # horizontal EPI: source row y, positions x_s + du * D_x
        pos_h = xs.view(1,1,W) + du * dx[:, rows_t]                                  # [B,R,W]
        cand_x = self._nearest_1d(pos_h, valid[:, rows_t], cols_t.to(img.dtype), K)  # [B,R,C,K]
        # vertical EPI: source column x, positions y_s + dv * D_y
        pos_v = ys.view(1,1,H) + dv * dy[:, :, cols_t].transpose(1, 2)               # [B,C,H]
        cand_y = self._nearest_1d(pos_v, valid[:, :, cols_t].transpose(1, 2), rows_t.to(img.dtype), K)
        cand_y = cand_y.transpose(1, 2)                                              # [B,R,C,K]

        # K x K combinations
        cy = cand_y[..., :, None].expand(-1,-1,-1,K,K).reshape(*cand_y.shape[:3], K*K)
        cx = cand_x[..., None, :].expand(-1,-1,-1,K,K).reshape(*cand_x.shape[:3], K*K)
        flat = cy * W + cx                                                           # [B,R,C,K*K]
        g = lambda t: torch.gather(t.reshape(B, 1, 1, H*W).expand(-1, len(rows), len(cols), -1), 3, flat)
        off_x = cx.to(img.dtype) + du * g(dx) - cols_t.view(1,1,-1,1).to(img.dtype)
        off_y = cy.to(img.dtype) + dv * g(dy) - rows_t.view(1,-1,1,1).to(img.dtype)
        dist = off_x.abs() + off_y.abs()
        dist = torch.where(g(valid) > 0, dist, torch.full_like(dist, float('inf')))
        best = torch.topk(dist, K, dim=-1, largest=False)
        flat = torch.gather(flat, 3, best.indices)
        offset = torch.stack([torch.gather(off_x, 3, best.indices), torch.gather(off_y, 3, best.indices)], -1)
        return flat, offset, best.values

    # ------------------------------------------------------------------ rendering
    def render(self, sources, u, v, rows=None, cols=None, chunk=8192):
        '''
        Renders the target view (u, v) (view units, relative to the light field grid) on the
        target pixels rows x cols from a list of sources (see make_source).
        returns rgb [B,3,R,C], disparity [B,1,R,C], weight entropy [B,1,R,C],
                min re-projection distance [B,1,R,C], weights [B*R*C,K]
        '''
        B, _, H, W = sources[0]['img'].shape
        rows = list(range(H)) if rows is None else rows
        cols = list(range(W)) if cols is None else cols
        R, C, K = len(rows), len(cols), self.num_related
        device, dtype = sources[0]['img'].device, sources[0]['img'].dtype

        # candidates from every source, then the K nearest overall
        flats, offs, dists = [], [], []
        for n, src in enumerate(sources):
            flat, off, dist = self.find_candidates(src, u, v, rows, cols)
            flats.append(flat + n*H*W)
            offs.append(off)
            dists.append(dist)
        flat, off, dist = torch.cat(flats, -1), torch.cat(offs, -2), torch.cat(dists, -1)
        if len(sources) > 1:
            best = torch.topk(dist, K, dim=-1, largest=False)
            flat = torch.gather(flat, 3, best.indices)
            off = torch.gather(off, 3, best.indices[..., None].expand(-1,-1,-1,-1,2))
            dist = best.values

        # per-pixel source features, stacked over sources: [B, Ctot, N*H*W]
        feat = torch.cat([torch.cat([s['img'], s['disp_x'], s['disp_y'], s['prompt']], 1).flatten(2)
                          for s in sources], -1)
        cams = torch.cat([torch.tensor([[s['u'], s['v']]], device=device, dtype=dtype).expand(H*W, 2)
                          for s in sources], 0)                                      # [N*H*W, 2]

        flat = flat.reshape(B, R*C*K)
        feat = torch.gather(feat, 2, flat[:, None].expand(-1, feat.shape[1], -1))     # [B,Ctot,R*C*K]
        feat = feat.permute(0, 2, 1).reshape(B*R*C, K, -1)
        color, sdx, sdy, prompt = feat[..., :3], feat[..., 3:4], feat[..., 4:5], feat[..., 5:]
        off = off.reshape(B*R*C, K, 2)

        # source rays
        pix = flat % (H*W)
        src_y = (pix // W).to(dtype).reshape(B*R*C, K)
        src_x = (pix % W).to(dtype).reshape(B*R*C, K)
        cam = cams[flat.reshape(-1)].reshape(B*R*C, K, 2)
        r_src = self.plucker(src_x, src_y, cam[..., 0], cam[..., 1], H, W)
        # target rays
        ty, tx = torch.meshgrid(torch.as_tensor(rows, device=device, dtype=dtype),
                                torch.as_tensor(cols, device=device, dtype=dtype), indexing='ij')
        tx = tx.reshape(1, -1).expand(B, -1).reshape(-1)
        ty = ty.reshape(1, -1).expand(B, -1).reshape(-1)
        r_tgt = self.plucker(tx, ty, torch.full_like(tx, u), torch.full_like(tx, v), H, W)

        cues = posenc(torch.cat([sdx, sdy, off], -1) / self.disp_norm, 1, self.pe_deg, torch_channel=False)

        weights = []
        for i in range(0, B*R*C, chunk):
            key = self.phi_k(torch.cat([r_src[i:i+chunk], prompt[i:i+chunk]], -1))      # Eq. 8
            tok = torch.cat([r_tgt[i:i+chunk, None].expand(-1, K, -1), key, cues[i:i+chunk]], -1)  # Eq. 9
            weights.append(self.transformer(tok))
        w = torch.cat(weights, 0)                                                     # [B*R*C, K]

        rgb = torch.sum(w[..., None] * color, 1).reshape(B, R, C, 3).permute(0, 3, 1, 2)
        disp = torch.sum(w[..., None] * sdx, 1).reshape(B, R, C, 1).permute(0, 3, 1, 2)
        entropy = torch.sum(-w * torch.log2(w + 1e-10), -1).reshape(B, R, C, 1).permute(0, 3, 1, 2)
        min_dist = dist[..., 0].reshape(B, R, C, 1).permute(0, 3, 1, 2)
        return rgb, disp, entropy, min_dist, w
