import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional
from torch import Tensor

from utils.utils import compute_iou, compute_effective_iou, xcycwh2xyxy


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
                 coord_criterion,
                 conf_criterion,
                 img_size,
                 grid_size,
                 lambda_xy=1,
                 lambda_wh=1,
                 lambda_obj=1,
                 lambda_noobj=1
                 ):
        super(YOLOLoss, self).__init__()
        self.coord_criterion = coord_criterion
        self.conf_criterion = conf_criterion
        self.img_size = img_size
        self.grid_size = grid_size
        self.lambda_xy = lambda_xy
        self.lambda_wh = lambda_wh
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

        self.cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]

    def forward(self, input, target):
        '''
        Compute standard YOLOv3 loss
        ```{input, target}[i, j, k, :] = (c, x, y, w, h)```
        :param input (torch.Tensor[N, S, S, 5]): raw model output (logits)
        :param target (torch.Tensor[N, S, S, 5]): yolo target
        :return:
        '''

        obj_mask = target[..., 0] > 0.
        noobj_mask = ~obj_mask

        obj_pred, noobj_pred = input[obj_mask], input[noobj_mask]
        obj_target, noobj_target = target[obj_mask], target[noobj_mask]

        noobj_pred_logit = noobj_pred[..., 0]
        noobj_target_conf = noobj_target[..., 0]
        loss_noobj = self.conf_criterion(noobj_pred_logit, noobj_target_conf)

        pred_xy = torch.sigmoid(obj_pred[..., 1:3])
        target_xy = obj_target[..., 1:3]
        loss_xy = self.coord_criterion(pred_xy, target_xy)

        pred_wh = torch.sigmoid(obj_pred[..., 3:])
        target_wh = obj_target[..., 3:]
        loss_wh = self.coord_criterion(pred_wh, target_wh)

        obj_pred_logit = obj_pred[..., 0]
        obj_target_conf = obj_target[..., 0]

        loss_obj = self.conf_criterion(obj_pred_logit, obj_target_conf)

        loss = self.lambda_xy * loss_xy + self.lambda_wh * loss_wh + \
               self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj

        return {
            'loss': loss,
            'loss_noobj': loss_noobj.detach(),
            'loss_xy': loss_xy.detach(),
            'loss_wh': loss_wh.detach(),
            'loss_obj': loss_obj.detach()
        }


class IoULoss(nn.Module):
    def __init__(self, reduction='none'):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute IoU loss
        :param input (torch.Tensor[N, 4]): predicted bounding boxes in ``(xc, yc, w, h)`` format
        :param target (torch.Tensor[N, 4]): target bounding boxes in ``(xc, yc, w, h)`` format
        """
        iou = compute_iou(bboxes1=input, bboxes2=target, bbox_transform=xcycwh2xyxy)
        if self.reduction == 'mean':
            iou = iou.mean()
        elif self.reduction == 'sum':
            iou = iou.sum()

        return 1 - iou


class EIoULoss(nn.Module):
    def __init__(self, reduction='none'):
        super(EIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute effective IoU loss
        :param input (torch.Tensor[N, 4]): predicted bounding boxes in ``(xc, yc, w, h)`` format
        :param target (torch.Tensor[N, 4]): target bounding boxes in ``(xc, yc, w, h)`` format
        """
        eiou = compute_effective_iou(bboxes1=input, bboxes2=target, bbox_transform=xcycwh2xyxy)
        if self.reduction == 'mean':
            eiou = eiou.mean()
        elif self.reduction == 'sum':
            eiou = eiou.sum()

        return 1 - eiou


class CustomYOLOLoss(nn.Module):
    def __init__(self,
                 bbox_criterion,
                 conf_criterion,
                 img_size,
                 grid_size,
                 lambda_noobj=1,
                 lambda_bbox=1,
                 lambda_obj=1
                 ):
        super(CustomYOLOLoss, self).__init__()
        self.bbox_criterion = bbox_criterion
        self.conf_criterion = conf_criterion
        self.img_size = img_size
        self.grid_size = grid_size
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj

        self.cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]

    def forward(self, input, target):
        '''
        Compute YOLOv3 loss with {G, C, E}IoULoss for bbox predictions
        ```{input, target}[i, j, k, :] = (c, x, y, w, h)```
        :param input (torch.Tensor[N, S, S, 5]): raw model output (logits)
        :param target (torch.Tensor[N, S, S, 5]): yolo target
        :return:
        '''

        obj_mask = target[..., 0] > 0.
        noobj_mask = ~obj_mask

        obj_pred, noobj_pred = input[obj_mask], input[noobj_mask]
        obj_target, noobj_target = target[obj_mask], target[noobj_mask]

        noobj_pred_logit = noobj_pred[..., 0]
        noobj_target_conf = noobj_target[..., 0]
        loss_noobj = self.conf_criterion(noobj_pred_logit, noobj_target_conf)

        pred_bbox = obj_pred[..., 1:]
        target_bbox = obj_target[..., 1:]
        loss_bbox = self.bbox_criterion(pred_bbox, target_bbox)

        obj_pred_logit = obj_pred[..., 0]
        obj_target_conf = obj_target[..., 0]

        loss_obj = self.conf_criterion(obj_pred_logit, obj_target_conf)

        loss = self.lambda_bbox * loss_bbox + \
               self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj

        return {
            'loss': loss,
            'loss_bbox': loss_bbox.detach(),
            'loss_conf': loss_obj.detach()
        }
