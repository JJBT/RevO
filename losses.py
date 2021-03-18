import torch
from torch import nn
from torch import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma):
        super(BinaryFocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, input, target):
        """
        Args:
             input (torch.Tensor): unnormalised model scores
             target (torch.Tensor): binary target
        """
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce)
        pt, target = pt.view(-1), target.view(-1)

        fl = self.pos_weight * (1 - pt) ** self.gamma * pt
        fl = torch.where(target, self.pos_weight * fl, (1 - self.pos_weight) * fl)
        return fl.mean()
