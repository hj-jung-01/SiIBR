import os
import random
import numpy as np
import skimage.io as skio
import torch
from torch.utils.data import Dataset

from data.data_aug import data_augmentation

# 4D light field loader for SiIBR training (Pov-ray dataset of Heber et al.).
# View (u, v) of an AN x AN light field is read from scene{u*AN+v:03d}.png, where u is the
# horizontal axis (stride AN in the file index) and v the vertical one.
# The first `num_test` scenes (sorted by name) are held out for testing.
# Each sample holds the center view (input), the views on the center row (for EPI losses)
# and one random view of the 4D grid.


def get_scene_list(lfinpath):
    return sorted([f for f in os.listdir(lfinpath) if '.' not in f and 'scene' in f])


class LF4DDataset(Dataset):
    def __init__(self, args, num_test=100, train=True):
        self.args = args
        scenes = get_scene_list(args.lfinpath)
        self.id_list = scenes[num_test:] if train else scenes[:num_test]
        self.train = train
        self.AN = args.angular_resolution
        self.crop_size = [args.crop_height, args.crop_width]

    def __len__(self):
        return len(self.id_list)

    def read_view(self, lf_path, u, v, seeds):
        img = skio.imread(os.path.join(lf_path, f'scene{str(u*self.AN+v).zfill(3)}.png')).astype('float32')[...,:3] / 255.
        img = data_augmentation(img, mode='image', seeds=seeds, crop_size=self.crop_size)
        return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,0,1)))).float()

    def get_random_seeds(self, H, W):
        if not self.train:
            return [0., 0., 1., 0., 0, 0, (H-self.crop_size[0])//2, (W-self.crop_size[1])//2]
        return [random.uniform(-0.15,0.15), random.uniform(0.,0.2),
                random.uniform(0.7, 1), random.uniform(-0.1,0.1), 0, 0,
                random.randint(0, H-self.crop_size[0]),
                random.randint(0, W-self.crop_size[1])]

    def __getitem__(self, idx):
        lf_path = os.path.join(self.args.lfinpath, self.id_list[idx])
        c = self.AN // 2
        seeds = self.get_random_seeds(self.args.img_height, self.args.img_width)

        center = self.read_view(lf_path, c, c, seeds)
        row = torch.stack([self.read_view(lf_path, u, c, seeds) for u in range(self.AN)], 1)  # [3,AN,h,w]
        u_r, v_r = random.randrange(self.AN), random.randrange(self.AN)
        view = self.read_view(lf_path, u_r, v_r, seeds)
        return center, row, view, torch.tensor([u_r - c, v_r - c], dtype=torch.float32)
