import torch
import torch.nn as nn
import torch.nn.functional as F


# Pixel-aligned SR prompt tokens (Eq. 7)
#   p_i = phi_p[ |grad I_src|_i, angle(grad I_src)_i, P_lo(i), P_hi(i) ]
# The low/high-frequency prompt modules follow the LPM/HPM of PromptRestorer:
# LPM modulates features in the frequency domain, HPM gates
# high-frequency residuals with local spatial interactions.


def image_gradient(img):
    # img: [B,3,H,W] -> magnitude [B,1,H,W], orientation as (cos, sin) [B,2,H,W]
    gray = 0.299*img[:,0:1] + 0.587*img[:,1:2] + 0.114*img[:,2:3]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device, dtype=img.dtype).view(1,1,3,3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(F.pad(gray, (1,1,1,1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(gray, (1,1,1,1), mode='replicate'), ky)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-12)
    ang = torch.atan2(gy, gx)
    return mag, torch.cat([torch.cos(ang), torch.sin(ang)], 1)


class LowFreqPrompt(nn.Module):
    # frequency-domain prompting: learnable per-channel spectral gain on a low-pass band
    def __init__(self, ch, radius=0.25):
        super(LowFreqPrompt, self).__init__()
        self.embed = nn.Conv2d(3, ch, 3, 1, 1)
        self.gain = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.radius = radius
        self.out = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        f = self.embed(x)
        H, W = f.shape[-2:]
        spec = torch.fft.rfft2(f, norm='ortho')
        fy = torch.fft.fftfreq(H, device=f.device).abs()[:, None]
        fx = torch.fft.rfftfreq(W, device=f.device)[None, :]
        low = ((fy**2 + fx**2).sqrt() < self.radius).to(f.dtype)
        spec = spec * (1 + low*(self.gain-1)) + low*self.bias
        f = torch.fft.irfft2(spec, s=(H, W), norm='ortho')
        return self.out(f)


class HighFreqPrompt(nn.Module):
    # spatial-domain gating of the high-frequency residual
    def __init__(self, ch):
        super(HighFreqPrompt, self).__init__()
        self.embed = nn.Conv2d(3, ch, 3, 1, 1)
        self.local = nn.Conv2d(ch, ch, 5, 1, 2, groups=ch)
        self.gate = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.Sigmoid())
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        f = self.embed(x)
        high = f - F.avg_pool2d(f, 5, 1, 2, count_include_pad=False)
        high = self.local(high)
        return self.out(high * self.gate(f))


class SRPromptEncoder(nn.Module):
    def __init__(self, prompt_dim=64, freq_dim=32):
        super(SRPromptEncoder, self).__init__()
        self.lpm = LowFreqPrompt(freq_dim)
        self.hpm = HighFreqPrompt(freq_dim)
        # phi_p: lightweight projection of the concatenated cues
        self.phi_p = nn.Sequential(nn.Conv2d(3 + 2*freq_dim, prompt_dim, 1), nn.GELU(),
                                   nn.Conv2d(prompt_dim, prompt_dim, 1))

    def forward(self, img_hr, img_lr=None):
        '''
        img_hr: (bicubic) up-sampled source image [B,3,sH,sW]
        img_lr: low-resolution source image [B,3,H,W] (LPM/HPM run at this resolution
                and are resized to the high-resolution grid); defaults to img_hr
        '''
        if img_lr is None:
            img_lr = img_hr
        mag, ori = image_gradient(img_hr)
        size = img_hr.shape[-2:]
        p_lo = F.interpolate(self.lpm(img_lr), size, mode='bilinear', align_corners=False)
        p_hi = F.interpolate(self.hpm(img_lr), size, mode='bilinear', align_corners=False)
        return self.phi_p(torch.cat([mag, ori, p_lo, p_hi], 1))  # [B,P,sH,sW]
