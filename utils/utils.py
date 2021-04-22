import time
import torch
import numpy as np
from collections import MutableMapping
import random
import pydoc
from omegaconf.dictconfig import DictConfig


def object_from_dict(d, parent=None, **default_kwargs):
    assert isinstance(d, (dict, DictConfig)) and 'type' in d
    kwargs = d.copy()
    kwargs = dict(kwargs)
    object_type = kwargs.pop('type')

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, (dict, DictConfig)) and 'type' in value:
            value = object_from_dict(value)
            kwargs[key] = value

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, '', sep=sep).items())  # GOD HELP ME
        else:
            value = v.item() if isinstance(v, torch.Tensor) else v
            items.append((new_key, value))

    return dict(items)


def loss_to_dict(loss):
    if not isinstance(loss, dict):
        return {'loss': loss}
    else:
        return loss


def set_determenistic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def data2device(data, device):
    return data.to(device)


def get_state_dict(model):
    if model is None:
        return None
    else:
        return unwrap_model(model).state_dict()


def load_state_dict(model, state_dict):
    if model is None:
        return None
    else:
        return model.load_state_dict(state_dict)


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def compute_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, bbox_transform=lambda x: x):
    '''
    Compute IoU between corresponding boxes of bboxes1 and bboxes2.

    :param bboxes1 (torch.Tensor[N, 4]):
    :param bboxes2 (torch.Tensor[N, 4]):
    :param bbox_transform (func): box coordinates converter to ``(xl, yt, xr, yb)`` format
    :return torch.Tensor[N, 1]:
    '''

    bboxes1 = bbox_transform(bboxes1)
    bboxes2 = bbox_transform(bboxes2)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter

    return inter / union


def _smallest_enclosing_box(bboxes1: torch.Tensor, bboxes2: torch.Tensor, bbox_transform=lambda x: x):
    bboxes1 = bbox_transform(bboxes1)
    bboxes2 = bbox_transform(bboxes2)

    lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    box = torch.cat([lt, rb], dim=1)
    return box


def compute_effective_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, bbox_transform=lambda x: x, pc_iou=None):
    '''
    Compute IoU between corresponding boxes of bboxes1 and bboxes2.

    :param bboxes1 (torch.Tensor[N, 4]):
    :param bboxes2 (torch.Tensor[N, 4]):
    :param bbox_transform (func): box coordinates converter to ``(xl, yt, xr, yb)`` format
    :param pc_iou (torch.Tensor[N, 1]): precomputed IoU
    :return torch.Tensor[N, 1]:
    '''

    bboxes1 = bbox_transform(bboxes1)
    bboxes2 = bbox_transform(bboxes2)

    iou = compute_iou(bboxes1, bboxes2) if pc_iou is None else pc_iou

    xcyc1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2
    xcyc2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2
    se_box = _smallest_enclosing_box(bboxes1, bboxes2)
    norm = torch.linalg.norm(xcyc2 - xcyc1, dim=-1)
    se_diag = torch.linalg.norm(se_box[:, 2:] - se_box[:, :2], dim=-1)
    dis = norm.square() / se_diag.square()

    wh1 = bboxes1[:, 2:] - bboxes1[:, :2]
    wh2 = bboxes2[:, 2:] - bboxes2[:, :2]
    se_box_wh = se_box[:, 2:] - se_box[:, :2]
    asp = ((wh1 - wh2).square() / se_box_wh.square()).sum(dim=-1)

    return iou - dis - asp

