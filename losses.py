import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=.5, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
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

        fl = self.alpha * (1 - pt) ** self.gamma * pt
        fl = torch.where(target.bool(), self.alpha * fl, (1 - self.alpha) * fl)
        return fl.mean()
