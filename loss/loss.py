import torch.nn as nn
import torch

from utils.calcST import calc_structure_tensor

# Weight(output from ray transformer) entropy loss
class EntropyLoss(nn.Module):
    def __init__(self, type='log2'):
        super(EntropyLoss, self).__init__()
        self.type = type

    def forward(self, prob):
        if self.type == 'log2':
            return torch.mean(torch.sum(-1*prob*torch.log2(prob+1e-10),-2))
        elif self.type == '1-p':
            return torch.mean(torch.sum(prob*torch.log2(1-prob),-2))

# structure tensor loss for EPIs
class StructureTensorLoss(nn.Module):
    def __init__(self, ksize=(3,3), sigma=(1.,1.)):
        super(StructureTensorLoss, self).__init__()

        self.ksize = ksize
        self.sigma = sigma
        self.MAE = nn.L1Loss()

    def forward(self, gt_epi, pred_epi):
        # taking [B, 3, H, W] as input

        #structure tensor
        st_gt = torch.cat(calc_structure_tensor(gt_epi, self.ksize, self.sigma), dim=1)
        st_pred = torch.cat(calc_structure_tensor(pred_epi, self.ksize, self.sigma), dim=1)

        return self.MAE(st_gt, st_pred)
