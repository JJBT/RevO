import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        Args:
             input (torch.Tensor): logits
             target (torch.Tensor): binary target
        """
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce)
        bce, pt, target = bce.view(-1), pt.view(-1), target.view(-1)

        fl = (1 - pt) ** self.gamma * bce
        if self.alpha is not None:
            fl = torch.where(target.bool(), self.alpha * fl, (1 - self.alpha) * fl)
        return fl.mean()
