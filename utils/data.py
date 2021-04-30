import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict

import torch
from torchvision.ops import box_convert


def get_coco_img_ids(coco):
    """
    Return a list of image ids according to annotations
    (use in case annotations were changed after coco loading).
    """
    img_ids = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_ids.add(ann['image_id'])

    return list(img_ids)


def load_coco_samples(coco):
    samples = []

    img_ids = get_coco_img_ids(coco)
    for img_meta in coco.loadImgs(img_ids):
        image_id = img_meta['id']
        file_name = img_meta['file_name']
        height = img_meta['height']
        width = img_meta['width']
        anns = coco.loadAnns(coco.getAnnIds(image_id))

        sample = {
            'image_id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'anns': anns
        }
        samples.append(sample)

    return samples


def save_coco_anns(anns, filename_to_save):
    with open(filename_to_save, 'w') as file:
        json.dump(anns, file)


def check_bbox_validity(bbox, format='coco'):
    if format == 'coco':
        is_valid = True
        if bbox[0] < 0 or bbox[1] < 0:
            is_valid = False

        if bbox[1] <= 0 or bbox[2] <= 0:
            is_valid = False

    else:
        raise NotImplementedError('unknown bbox format')

    return is_valid


def get_bbox_scale(coco, ann):
    image_id = ann['image_id']
    bbox = ann['bbox']

    image = coco.loadImgs(ids=[image_id])[0]
    width, height = image['width'], image['height']

    x_scale = round(bbox[2] / width, 2)
    y_scale = round(bbox[3] / height, 2)

    return x_scale, y_scale


def get_category_based_anns(coco):
    coco_samples = load_coco_samples(coco)

    category_based_anns = []

    for sample in coco_samples:
        file_name = sample['file_name']
        anns = sample['anns']

        category_dict = defaultdict(list)
        for ann in anns:
            ann.pop("segmentation", None)
            ann.pop("keypoints", None)

            category_id = ann['category_id']
            category_dict[category_id].append(ann)

        for key, item in category_dict.items():
            instance_ann = {
                'image_id': sample['image_id'],
                'file_name': file_name,
                'anns': item
                }
            category_based_anns.append(instance_ann)

    return category_based_anns


def get_kps_set2idx(anns, idx2kps):
    kps_sets = set()
    for ann in anns:
        kps_visibility = ann['keypoints'][2::3]
        kps_set = frozenset(idx2kps[i] for i, v in enumerate(kps_visibility) if v == 2)
        kps_sets.add(kps_set)

    return {kps_set: i for i, kps_set in enumerate(list(kps_sets))}


def get_anns_info_df(coco, save=None):
    anns = coco.loadAnns(coco.getAnnIds())
    cats = coco.cats

    anns_info = defaultdict(list)
    for i, ann in enumerate(anns):
        id = ann['id']
        image_id = ann['image_id']
        is_crowd = ann['iscrowd']
        bbox = ann['bbox']

        anns_info['id'].append(id)
        anns_info['image_id'].append(image_id)
        anns_info['category_id'].append(ann['category_id'])
        anns_info['category'].append(cats[ann['category_id']]['name'])
        anns_info['is_crowd'].append(is_crowd)

        if not is_crowd:
            anns_info['bbox_x'].append(bbox[0])
            anns_info['bbox_y'].append(bbox[1])
            anns_info['bbox_width'].append(bbox[2])
            anns_info['bbox_height'].append(bbox[3])

            x_scale, y_scale = get_bbox_scale(coco, ann)
            anns_info['bbox_x_scale'].append(x_scale)
            anns_info['bbox_y_scale'].append(y_scale)

        else:
            anns_info['bbox_x'].append(-1)
            anns_info['bbox_y'].append(-1)
            anns_info['bbox_width'].append(-1)
            anns_info['bbox_height'].append(-1)

            anns_info['bbox_x_scale'].append(-1)
            anns_info['bbox_y_scale'].append(-1)

    anns_info = pd.DataFrame(anns_info)

    if save:
        anns_info.to_csv(os.path.join('.', f'data/{save}.csv'))

    return anns_info


def thr_confidence(input, thr):
    new_input = input.clone()
    new_input[..., 0] = (new_input[...,  0] > thr).float()
    return new_input


def get_relative_coords(bbox, img_size, grid_size):
    cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]
    bbox = xywh2xcycwh(bbox)
    x, y = (bbox[0] % cell_size[0]) / cell_size[0], (bbox[1] % cell_size[1]) / cell_size[1]
    w, h = bbox[2] / img_size[0], bbox[3] / img_size[1]
    return [x, y, w, h]


def to_yolo_target(bboxes, img_size, stride):
    w, h = img_size
    grid_w, grid_h = w // stride, h // stride
    cell_w, cell_h = w // grid_w, h // grid_h

    target = np.zeros((grid_h, grid_w, 5), dtype=np.float)
    for bbox in bboxes:
        img_xc, img_yc = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        cell_x, cell_y = int(img_xc / cell_w), int(img_yc / cell_h)

        cell_target = [1., *get_relative_coords(bbox, img_size, (grid_w, grid_h))]
        target[cell_y, cell_x] = np.array(cell_target)

    return target


def from_yolo_target(target, img_size, grid_size):
    """Returns list of bboxes in ``(xl, yt, w, h)`` format"""
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]
    new_target = target.copy()

    x_offset = np.expand_dims(np.arange(grid_size[1]), axis=0).repeat(grid_size[0], axis=0) * cell_size[0]
    new_target[:, :, 1] = x_offset + new_target[:, :, 1] * cell_size[0]
    y_offset = np.expand_dims(np.arange(grid_size[0]), axis=0).T.repeat(grid_size[1], axis=1) * cell_size[1]
    new_target[:, :, 2] = y_offset + new_target[:, :, 2] * cell_size[1]

    new_target[:, :, 3] = new_target[:, :, 3] * img_size[0]
    new_target[:, :, 4] = new_target[:, :, 4] * img_size[1]

    new_target[:, :, 1] -= new_target[:, :, 3] // 2
    new_target[:, :, 2] -= new_target[:, :, 4] // 2

    new_target = new_target[new_target[:, :, 0] > 0][:, 1:].tolist()

    return new_target


def from_yolo_target_torch(target, img_size, grid_size):
    """Returns bboxes in ``(c, xl, yt, w, h)`` format"""

    if torch.is_tensor(target):
        target = target.detach().cpu()

    cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]
    new_target = target.clone()
    new_target = torch.stack(torch.split(new_target, 5, -1))

    x_offset = torch.repeat_interleave(torch.unsqueeze(torch.arange(grid_size[1]), dim=0), repeats=grid_size[0], dim=0) * cell_size[0]
    new_target[..., 1] = x_offset + new_target[..., 1] * cell_size[0]
    y_offset = torch.repeat_interleave(torch.unsqueeze(torch.arange(grid_size[0]), dim=0).T, repeats=grid_size[1], dim=1) * cell_size[1]
    new_target[..., 2] = y_offset + new_target[..., 2] * cell_size[1]

    new_target[..., 3] = new_target[..., 3] * img_size[0]
    new_target[..., 4] = new_target[..., 4] * img_size[1]

    new_target[..., 1] -= new_target[..., 3] // 2
    new_target[..., 2] -= new_target[..., 4] // 2

    new_target = new_target[new_target[..., 0] > 0]
    return new_target


def xcycwh2xyxy(bboxes):
    if torch.is_tensor(bboxes):
        return box_convert(bboxes, in_fmt='cxcywh', out_fmt='xyxy')

    elif isinstance(bboxes, np.ndarray):
        xc, yc, w, h = bboxes.T
        x1 = xc - 0.5 * w
        y1 = yc - 0.5 * h
        x2 = xc + 0.5 * w
        y2 = yc + 0.5 * h
        return np.stack([x1, y1, x2, y2], axis=-1)
    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            xc, yc, w, h = bboxes
            x1 = xc - 0.5 * w
            y1 = yc - 0.5 * h
            x2 = xc + 0.5 * w
            y2 = yc + 0.5 * h
            return [x1, y1, x2, y2]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                xc, yc, w, h = bbox
                x1 = xc - 0.5 * w
                y1 = yc - 0.5 * h
                x2 = xc + 0.5 * w
                y2 = yc + 0.5 * h
                new_bboxes.append([x1, y1, x2, y2])
            return new_bboxes


def xyxy2xcycwh(bboxes):
    if torch.is_tensor(bboxes):
        return box_convert(bboxes, in_fmt='xyxy', out_fmt='cxcywh')

    elif isinstance(bboxes, np.ndarray):
        x1, y1, x2, y2 = bboxes.T
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.stack((cx, cy, w, h), axis=-1)

    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            x1, y1, x2, y2 = bboxes
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            return [cx, cy, w, h]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                new_bboxes.append([cx, cy, w, h])
            return new_bboxes


def xywh2xcycwh(bboxes):
    if torch.is_tensor(bboxes):
        x, y, w, h = bboxes.unbind(-1)
        xc = x + 0.5 * w
        yc = y + 0.5 * h
        return torch.stack([xc, yc, w, h], dim=-1)

    elif isinstance(bboxes, np.ndarray):
        x, y, w, h = bboxes.T
        xc = x + 0.5 * w
        yc = y + 0.5 * h
        return np.stack((xc, yc, w, h), axis=-1)

    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            x, y, w, h = bboxes
            xc = x + 0.5 * w
            yc = y + 0.5 * h
            return [xc, yc, w, h]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                xc = x + 0.5 * w
                yc = y + 0.5 * h

                new_bboxes.append([xc, yc, w, h])
            return new_bboxes


def xcycwh2xywh(bboxes):
    if torch.is_tensor(bboxes):
        xc, yc, w, h = bboxes.unbind(-1)
        x = xc - 0.5 * w
        y = yc - 0.5 * h
        return torch.stack([x, y, w, h], dim=-1)

    elif isinstance(bboxes, np.ndarray):
        xc, yc, w, h = bboxes.T
        x = xc - 0.5 * w
        y = yc - 0.5 * h
        return np.stack((x, y, w, h), axis=-1)

    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            xc, yc, w, h = bboxes
            x = xc - 0.5 * w
            y = yc - 0.5 * h
            return [x, y, w, h]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                xc, yc, w, h = bbox
                x = xc - 0.5 * w
                y = yc - 0.5 * h

                new_bboxes.append([x, y, w, h])
            return new_bboxes


def xywh2xyxy(bboxes):
    if torch.is_tensor(bboxes):
        return box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')

    elif isinstance(bboxes, np.ndarray):
        x, y, w, h = bboxes.T
        return np.stack([x, y, x + w, y + h], axis=-1)

    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            x, y, w, h = bboxes
            return [x, y, x + w, y + h]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                new_bboxes.append([x, y, x + w, y + h])
            return new_bboxes


def xyxy2xywh(bboxes):
    if torch.is_tensor(bboxes):
        return box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh')

    elif isinstance(bboxes, np.ndarray):
        x1, y1, x2, y2 = bboxes.T
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)

    elif isinstance(bboxes, (list, tuple)):
        if isinstance(bboxes[0], (int, float)):
            x1, y1, x2, y2 = bboxes
            return [x1, y1, x2 - x1, y2 - y1]
        elif isinstance(bboxes[0], (list, tuple)):
            new_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                new_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            return new_bboxes