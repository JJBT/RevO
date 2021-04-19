import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional
from torch import Tensor


def binary_focal_loss_with_logits(input, target, gamma, alpha, pos_weight, reduction):
    bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(-bce)
    bce, pt, target = bce.view(-1), pt.view(-1), target.view(-1)

    fl = (1 - pt) ** gamma * bce
    if alpha is not None:
        fl = torch.where(target.bool(), alpha * fl, (1 - alpha) * fl)
    if pos_weight is not None:
        fl = torch.where(target.bool(), pos_weight * fl, fl)

    if reduction != 'none':
        try:
            fl = getattr(fl, reduction)()
        except AttributeError:
            raise AttributeError('Unknown reduction type')
    return fl


class BFLWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, pos_weight: Optional[Tensor] = None, reduction='mean'):
        super(BFLWithLogitsLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
             input (torch.Tensor): logits
             target (torch.Tensor): binary target
        """
        return binary_focal_loss_with_logits(input, target, self.gamma, self.alpha, self.pos_weight, self.reduction)


class YOLOLoss(nn.Module):
    def __init__(self,
                 bbox_criterion,
                 conf_criterion,
                 img_size,
                 grid_size,
                 ):
        super(YOLOLoss, self).__init__()
        self.bbox_criterion = bbox_criterion
        self.conf_criterion = conf_criterion
        self.img_size = img_size
        self.grid_size = grid_size

        self.cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]

    def forward(self, input, target):
        # target: (B, S, S, 5)
        # target[i, j, k, :] = [c, x, y, w, h]
        input = input.view(-1, *self.grid_size, 5)
        obj_mask = target[..., 0] > 0.
        noobj_mask = ~obj_mask

        obj_pred, noobj_pred = input[obj_mask], input[noobj_mask]
        obj_target, noobj_target = target[obj_mask], target[noobj_mask]

        # noobj_pred_logit = noobj_pred[..., 0]
        # noobj_target_conf = noobj_target[..., 0]
        # loss_noobj = self.conf_criterion(noobj_pred_logit, noobj_target_conf)

        pred_xy = torch.sigmoid(obj_pred[..., 1:3])
        target_xy = obj_target[..., 1:3]
        loss_xy = self.bbox_criterion(pred_xy, target_xy)

        pred_wh = torch.sigmoid(obj_pred[..., 3:])
        target_wh = obj_target[..., 3:]
        loss_wh = self.bbox_criterion(pred_wh, target_wh)

        # obj_pred_logit = obj_pred[..., 0]
        # obj_target_conf = obj_target[..., 0]
        pred_logit = input[..., 0]
        target_conf = target[..., 0]
        loss_conf = self.conf_criterion(pred_logit, target_conf)

        # loss = loss_xy + loss_wh + loss_conf + loss_noobj
        loss = loss_xy + loss_wh + loss_conf
        return loss
