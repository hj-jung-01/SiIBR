from __future__ import print_function
import numpy as np
import skimage.color as color

def hsv_aug(x, seeds):
    hsv = color.rgb2hsv(x)
    hsv[:,:,0] = hsv[:,:,0] + seeds[0]
    hsv[:,:,1] = hsv[:,:,1] + seeds[1]
    x = color.hsv2rgb(hsv)
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x

def constrast_aug(x, seeds):
    m = np.median(x)
    s = seeds[2]
    x = m+(x-m)*s
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x

def brightness_aug(x, seeds):
    x = x + seeds[3]
    x = np.minimum(x, 1)
    x = np.maximum(x, 0)
    return x

def horizontal_flip_aug(x, mode, direction, seeds):
    if seeds[4]:
        if mode == 'image':
            x = np.fliplr(x)
        elif mode == 'disparity' and direction == 'horizontal':
            x = -np.fliplr(x)
        elif mode == 'disparity' and direction == 'vertical':
            x = np.fliplr(x)
    return x

def vertical_flip_aug(x, mode, direction, seeds):
    if seeds[5]:
        if mode == 'image':
            x = np.flipud(x)
        elif mode == 'disparity' and direction == 'horizontal':
            x = np.flipud(x)
        elif mode == 'disparity' and direction == 'vertical':
            x = -np.flipud(x)
    return x

def gaussian_aug(x):
    temp = (np.max(x) - np.min(x)) * 255
    if temp > 1:
        standard = np.log2(temp) * 0.01
        x = x + np.random.normal(0, standard, x.shape)
        x = np.minimum(x, 1)
        x = np.maximum(x, 0)
        return x
    else:
        return x

def random_crop(x,seeds,size):
    return x[seeds[6]:seeds[6]+size[0],seeds[7]:seeds[7]+size[1]]

def data_augmentation(x, mode, seeds, crop_size, direction=None, bluropt = 0):
    #mode = 'image' or 'disparity'
    #direction = 'horizontal' or 'vertical' or None
    if mode == 'image':
        x = horizontal_flip_aug(x, mode, direction, seeds)
        x = vertical_flip_aug(x, mode, direction, seeds)
        x = hsv_aug(x, seeds)
        x = constrast_aug(x, seeds)
        x = brightness_aug(x, seeds)
        x = random_crop(x, seeds, crop_size)
        if bluropt:
            x = gaussian_aug(x)
    elif mode == 'disparity':
        x = horizontal_flip_aug(x, mode, direction, seeds)
        x = vertical_flip_aug(x, mode, direction, seeds)
        x = random_crop(x, seeds, crop_size)
    return x
