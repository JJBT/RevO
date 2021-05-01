import time

import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional
from torch import Tensor

from utils.utils import compute_iou, compute_effective_iou
from utils.data import xcycwh2xyxy


def binary_focal_loss_with_logits(input, target, gamma, alpha, pos_weight, reduction):
    bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(-bce)
    bce, pt, target = bce.view(-1), pt.view(-1), target.view(-1)

    fl = (1 - pt) ** gamma * bce
    if alpha is not None:
        fl = torch.where(target.bool(), alpha * fl, (1 - alpha) * fl)
    if pos_weight is not None:
        fl = torch.where(target.bool(), pos_weight * fl, fl)

    if reduction == 'mean':
        fl = fl.mean()
    elif reduction == 'sum':
        fl = fl.sum()

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
        iou_loss = 1 - iou
        if self.reduction == 'mean':
            iou_loss = iou_loss.mean()
        elif self.reduction == 'sum':
            iou_loss = iou_loss.sum()

        return iou_loss


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
        eiou_loss = 1 - eiou
        if self.reduction == 'mean':
            eiou_loss = eiou_loss.mean()
        elif self.reduction == 'sum':
            eiou_loss = eiou_loss.sum()

        return eiou_loss


class FocalEIoULoss(nn.Module):
    def __init__(self, gamma=2, reduction='none'):
        super(FocalEIoULoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute effective IoU focal loss
        :param input (torch.Tensor[N, 4]): predicted bounding boxes in ``(xc, yc, w, h)`` format
        :param target (torch.Tensor[N, 4]): target bounding boxes in ``(xc, yc, w, h)`` format
        """

        iou = compute_iou(bboxes1=input, bboxes2=target, bbox_transform=xcycwh2xyxy)
        eiou = compute_effective_iou(bboxes1=input, bboxes2=target, bbox_transform=xcycwh2xyxy, pc_iou=iou)

        eiou_fl = iou ** self.gamma * (1 - eiou)
        if self.reduction == 'mean':
            eiou_fl = eiou_fl.mean()
        elif self.reduction == 'sum':
            eiou_fl = eiou_fl.sum()

        return eiou_fl


class CustomYOLOLoss(nn.Module):
    def __init__(self,
                 bbox_criterion,
                 conf_criterion,
                 lambda_noobj=1,
                 lambda_bbox=1,
                 lambda_obj=1
                 ):
        super(CustomYOLOLoss, self).__init__()
        self.bbox_criterion = bbox_criterion
        self.conf_criterion = conf_criterion
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj


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

        n_bbox = input.shape[-1] // 5
        conf_idx = torch.arange(n_bbox * 5) % 5 == 0
        coord_idx = ~conf_idx

        obj_pred, noobj_pred = input[obj_mask], input[noobj_mask]
        obj_target, noobj_target = target[obj_mask], target[noobj_mask]

        # Cells with no objects
        noobj_pred_logit = noobj_pred[..., conf_idx]
        loss_noobj = self.conf_criterion(noobj_pred_logit, torch.zeros_like(noobj_pred_logit))

        # Cells with objects
        # Responsible bboxes
        pred_bbox = torch.sigmoid(
            obj_pred[..., coord_idx].view(obj_pred.shape[0] * n_bbox, 4)
        )
        target_bbox = obj_target[..., 1:]
        iou = compute_iou(
            pred_bbox,
            torch.repeat_interleave(target_bbox[:, None, :], repeats=n_bbox, dim=1).view(obj_target.shape[0] * n_bbox, 4),
            bbox_transform=xcycwh2xyxy
        )
        responsible_idx = iou.view(obj_target.shape[0], n_bbox).argmax(dim=1)
        pred_resp_bbox = torch.stack(
            [pred_bbox[n_bbox * i + s] for i, s in zip(range(obj_pred.shape[0]), responsible_idx)]
        )

        loss_bbox = self.bbox_criterion(
            pred_resp_bbox,
            target_bbox
        )

        # Responsible confs
        pred_logit = obj_pred[..., conf_idx]
        pred_resp_logit = torch.stack(
            [pred_logit[i, s] for i, s in zip(range(obj_pred.shape[0]), responsible_idx)]
        )
        loss_obj = self.conf_criterion(
            pred_resp_logit,
            torch.ones_like(pred_resp_logit)
        )

        # Not responsible confs
        idx = torch.repeat_interleave(torch.arange(n_bbox)[None, :], repeats=obj_pred.shape[0], dim=0).to(responsible_idx.device)
        not_responsible_idx = idx[idx != responsible_idx.unsqueeze(1)]
        if not_responsible_idx.nelement() > 0:
            pred_not_resp_logit = torch.stack(
                [pred_logit[i, s] for i, s in zip(torch.repeat_interleave(torch.arange(obj_pred.shape[0]), n_bbox - 1), not_responsible_idx)]
            )
            loss_noobj += self.conf_criterion(
                pred_not_resp_logit,
                torch.zeros_like(pred_not_resp_logit)
            )
        loss = self.lambda_obj * loss_obj + self.lambda_bbox * loss_bbox + \
               self.lambda_noobj * loss_noobj

        return {
            'loss': loss,
            'loss_noobj': loss_noobj.detach(),
            'loss_bbox': loss_bbox.detach(),
            'loss_obj': loss_obj.detach()
        }


class CustomLoss(torch.nn.Module):
    def __init__(self,
                 bbox_criterion,
                 conf_criterion,
                 lambda_noobj=1,
                 lambda_bbox=1,
                 lambda_obj=1
                 ):
        super(CustomLoss, self).__init__()
        self.bbox_criterion = bbox_criterion
        self.conf_criterion = conf_criterion
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj


    def forward(self, input, target):
        """Compute YOLO loss
        Args:
            pred: (Tensor) yolo output, sized [n_batch, Bx5 + C, S, S], 5=len([x, y, w, h, conf])
            target: (Tensor) targets, sized [n_batch, Bx5 + C, S, S]
        Returns:
            (Tensor) loss value, sized [1,]
        """

        n_bboxes = input.shape[-1] // 5
        obj_mask = target[..., 0] > 0
        n_obj_cells = obj_mask.sum()
        obj_pred = input[obj_mask].view(n_obj_cells * n_bboxes, 5)
        n_noobj_cells = input.shape[0] * input.shape[1] * input.shape[2] - n_obj_cells
        noobj_pred = input[~obj_mask].view(n_noobj_cells * n_bboxes, 5)

        obj_target = target[obj_mask]

        # Compute loss for cells which have no objects
        noobj_logit = noobj_pred[..., 0]
        loss_noobj = self.conf_criterion(noobj_logit, torch.zeros_like(noobj_logit))

        # Compute loss for cells which contain objects
        pred_bbox = torch.sigmoid(obj_pred[:, 1:])
        target_bbox = obj_target[:, 1:]

        iou = compute_iou(
            pred_bbox,
            torch.repeat_interleave(
                target_bbox[:, None, :], repeats=n_bboxes, dim=1
            ).view(n_obj_cells * n_bboxes, 4),
            bbox_transform=xcycwh2xyxy
        )
        responsible_idx = iou.view(n_obj_cells, n_bboxes).argmax(dim=1)
        obj_responsible_mask = torch.zeros(n_obj_cells * n_bboxes, dtype=torch.bool)
        for cell_idx, max_iou_idx in enumerate(responsible_idx):
            obj_responsible_mask[n_bboxes * cell_idx + max_iou_idx] = 1

        pred_responsible = obj_pred[obj_responsible_mask]

        if n_bboxes > 1:
            loss_noobj += self.conf_criterion(
                obj_pred[~obj_responsible_mask][..., 0],
                torch.zeros_like(obj_pred[~obj_responsible_mask][..., 0])
            )

        loss_bbox = self.bbox_criterion(torch.sigmoid(pred_responsible[..., 1:]), obj_target[..., 1:])
        loss_obj = self.conf_criterion(pred_responsible[..., 0], obj_target[..., 0])

        loss = self.lambda_obj * loss_obj + self.lambda_bbox * loss_bbox + \
               self.lambda_noobj * loss_noobj

        return {
            'loss': loss,
            'loss_noobj': loss_noobj.detach(),
            'loss_bbox': loss_bbox.detach(),
            'loss_obj': loss_obj.detach()
        }
