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
        :param target (torch.Tensor[N, S, S, n_bbox * 5]): yolo target
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
            torch.repeat_interleave(target_bbox[:, None, :], repeats=n_bbox, dim=1).view(obj_target.shape[0] * n_bbox,
                                                                                         4),
            bbox_transform=xcycwh2xyxy
        )
        resbonsible_idx = iou.view(obj_target.shape[0], n_bbox).argmax(dim=1)
        pred_resp_bbox = torch.stack(
            [pred_bbox[n_bbox * i + s] for i, s in zip(range(obj_pred.shape[0]), resbonsible_idx)]
        )
        loss_xy = self.coord_criterion(
            pred_resp_bbox[..., :2],
            target_bbox[..., :2]
        )

        loss_wh = self.coord_criterion(
            pred_resp_bbox[..., 2:],
            target_bbox[..., 2:]
        )

        # Responsible confs
        pred_logit = obj_pred[..., conf_idx]
        pred_resp_logit = torch.stack(
            [pred_logit[i, s] for i, s in zip(range(obj_pred.shape[0]), resbonsible_idx)]
        )
        loss_obj = self.conf_criterion(
            pred_resp_logit,
            torch.ones_like(pred_resp_logit)
        )

        # Not responsible confs
        not_responsible_idx = torch.repeat_interleave(torch.arange(n_bbox)[None, :], repeats=obj_pred.shape[0], dim=0)
        not_responsible_idx = not_responsible_idx[not_responsible_idx != resbonsible_idx.unsqueeze(1)]
        if not_responsible_idx.nelement() > 0:
            pred_not_resp_logit = torch.stack(
                [pred_logit[i, s] for i, s in
                 zip(torch.repeat_interleave(torch.arange(obj_pred.shape[0]), n_bbox - 1), not_responsible_idx)]
            )
            loss_noobj += self.conf_criterion(pred_not_resp_logit, torch.zeros_like(pred_not_resp_logit))


        loss = self.lambda_noobj * loss_noobj + self.lambda_xy * loss_xy + \
               self.lambda_wh * loss_wh + self.lambda_obj * loss_obj

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
        resbonsible_idx = iou.view(obj_target.shape[0], n_bbox).argmax(dim=1)
        pred_resp_bbox = torch.stack(
            [pred_bbox[n_bbox * i + s] for i, s in zip(range(obj_pred.shape[0]), resbonsible_idx)]
        )
        loss_bbox = self.bbox_criterion(
            pred_resp_bbox,
            target_bbox
        )

        # Responsible confs
        pred_logit = obj_pred[..., conf_idx]
        pred_resp_logit = torch.stack(
            [pred_logit[i, s] for i, s in zip(range(obj_pred.shape[0]), resbonsible_idx)]
        )
        loss_obj = self.conf_criterion(
            pred_resp_logit,
            torch.ones_like(pred_resp_logit)
        )

        # Not responsible confs
        not_responsible_idx = torch.repeat_interleave(torch.arange(n_bbox)[None, :], repeats=obj_pred.shape[0], dim=0)
        not_responsible_idx = not_responsible_idx[not_responsible_idx != resbonsible_idx.unsqueeze(1)]
        if not_responsible_idx.nelement() > 0:
            pred_not_resp_logit = torch.stack(
                [pred_logit[i, s] for i, s in zip(torch.repeat_interleave(torch.arange(obj_pred.shape[0]), n_bbox - 1), not_responsible_idx)]
            )
            loss_noobj += self.conf_criterion(pred_not_resp_logit, torch.zeros_like(pred_not_resp_logit))

        loss = self.lambda_obj * loss_obj + self.lambda_bbox * loss_bbox + \
               self.lambda_noobj * loss_noobj

        return {
            'loss': loss,
            'loss_noobj': loss_noobj.detach(),
            'loss_bbox': loss_bbox.detach(),
            'loss_obj': loss_obj.detach()
        }
