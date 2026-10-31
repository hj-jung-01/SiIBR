# Structure tensor calculator

import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Tuple

class RGB2Gray(nn.Module):
    def __init__(self):
        super(RGB2Gray, self).__init__()
        self.weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")
        gray = F.conv2d(x, self.weights.to(x))
        return gray

def calc_structure_tensor(img_tensor, kernel_size=(3,3), sigma=(1.,1.)):

    rgb2gray = RGB2Gray()
    gray_img = rgb2gray(img_tensor)
    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def sobel_filter(x):
        # x: (b,1,h,w), float32 or float64
        # dx, dy: (b,1,h,w)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        grad_orientation = torch.atan2(grad_y, grad_x)
        return grad_x, grad_y, grad_magnitude, grad_orientation

    def gaussian(window_size: int, sigma: float) -> torch.Tensor:
        device, dtype = None, torch.float32
        if isinstance(sigma, torch.Tensor):
            device, dtype = sigma.device, sigma.dtype
        x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        if window_size % 2 == 0:
            x = x + 0.5
        gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
        return gauss / gauss.sum()

    def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
        if (not isinstance(kernel_size, int) or (
                (kernel_size % 2 == 0) and not force_even) or (
                kernel_size <= 0)):
            raise TypeError(
                "kernel_size must be an odd positive integer. "
                "Got {}".format(kernel_size)
            )
        window_1d: torch.Tensor = gaussian(kernel_size, sigma)
        return window_1d

    def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(
                "kernel_size must be a tuple of length two. Got {}".format(
                    kernel_size
                )
            )
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError(
                "sigma must be a tuple of length two. Got {}".format(sigma)
            )
        ksize_x, ksize_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
        kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
        kernel_2d: torch.Tensor = torch.matmul(
            kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
        )
        return kernel_2d

    if not isinstance(kernel_size, tuple):
        k = kernel_size
        kernel_size = (k,k)
    g = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), 0)
    gaussian_kernel = []
    for _ in range(gray_img.shape[1]):
        gaussian_kernel.append(torch.unsqueeze(g, 0))
    gaussian_kernel = torch.cat(gaussian_kernel, dim = 1)
    dx, dy, _, orientation = sobel_filter(gray_img)
    gaussian_kernel = gaussian_kernel.to(gray_img)
    Jxx = F.conv2d(dx*dx, gaussian_kernel, padding = 'same')
    Jxy = F.conv2d(dx*dy, gaussian_kernel, padding = 'same')
    Jyy = F.conv2d(dy*dy, gaussian_kernel, padding = 'same')
    return Jxx, Jxy, Jyy
