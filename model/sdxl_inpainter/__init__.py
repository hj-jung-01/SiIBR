import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from diffusers import AutoPipelineForInpainting


# Training-free RGB inpainter for dis-occluded regions (SDXL-inpainting, inference only).
# The disparity of the filled region is estimated by the caller.
class Inpainter():
    def __init__(self,
                 device=torch.device('cuda'),
                 prompt="fill as background",
                 num_inference_steps=20,
                 guidance_scale=8.0,
                 seed=0):

        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(
                        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def __call__(self, img, mask):
        '''
        img: rendered RGB [1,3,H,W] in [0,1]
        mask: 1 for valid (rendered) pixels, 0 for dis-occluded pixels [1,1,H,W]
        '''
        _,_,H,W = img.shape
        inpainted = self.pipe(prompt=self.prompt,
                              image=img.half(),
                              mask_image=1-mask.half(),
                              guidance_scale=self.guidance_scale,
                              num_inference_steps=self.num_inference_steps,  # steps between 15 and 30 work well
                              strength=0.99,  # make sure to use `strength` below 1.0
                              generator=self.generator,
                             ).images[0]
        inpainted = self.to_tensor(inpainted).unsqueeze(0).float().to(self.device)
        inpainted = F.interpolate(inpainted, (H, W), None, 'bilinear', True, False, True)

        # keep the rendered pixels, fill only the dis-occluded region
        return img*mask + inpainted*(1-mask)
