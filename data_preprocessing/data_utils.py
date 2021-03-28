from pycocotools.coco import COCO
import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import sys


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


def get_kps_set2idx(anns, idx2kps):
    kps_sets = set()
    for ann in tqdm(anns):
        kps_visibility = ann['keypoints'][2::3]
        kps_set = frozenset(idx2kps[i] for i, v in enumerate(kps_visibility) if v == 2)
        kps_sets.add(kps_set)

    return {kps_set: i for i, kps_set in enumerate(list(kps_sets))}


def get_anns_info_df(coco, save=None):
    anns = coco.loadAnns(coco.getAnnIds())
    cats = coco.loadCats(coco.getCatIds())[0]
    kps2idx = {kp: idx for kp, idx in zip(cats['keypoints'], list(range(len(cats['keypoints']))))}
    idx2kps = {idx: kp for kp, idx in kps2idx.items()}
    kps_set2idx = get_kps_set2idx(anns, idx2kps)

    anns_info = defaultdict(list)
    for i, ann in tqdm(enumerate(anns)):
        id = ann['id']
        image_id = ann['image_id']
        is_crowd = ann['iscrowd']
        bbox = ann['bbox']
        num_keypoints = ann['num_keypoints']
        keypoints = ann['keypoints']

        anns_info['id'].append(id)
        anns_info['image_id'].append(image_id)
        anns_info['is_crowd'].append(is_crowd)

        if not is_crowd:
            anns_info['bbox_x'].append(bbox[0])
            anns_info['bbox_y'].append(bbox[1])
            anns_info['bbox_width'].append(bbox[2])
            anns_info['bbox_height'].append(bbox[3])

            x_scale, y_scale = get_bbox_scale(coco, ann)
            anns_info['bbox_x_scale'].append(x_scale)
            anns_info['bbox_y_scale'].append(y_scale)

            kps_visibility = keypoints[2::3]
            visible_kps = frozenset(idx2kps[i] for i, v in enumerate(kps_visibility) if v == 2)

            anns_info['num_keypoints'].append(len(visible_kps))
            anns_info['kps_set_idx'].append(kps_set2idx[visible_kps])
            anns_info['kps'].append(list(visible_kps))

        else:
            anns_info['bbox_x'].append(-1)
            anns_info['bbox_y'].append(-1)
            anns_info['bbox_width'].append(-1)
            anns_info['bbox_height'].append(-1)

            anns_info['bbox_x_scale'].append(-1)
            anns_info['bbox_y_scale'].append(-1)

            anns_info['num_keypoints'].append(-1)
            anns_info['kps_set_idx'].append(-1)
            anns_info['kps'].append(-1)

    anns_info = pd.DataFrame(anns_info)

    if save:
        anns_info.to_csv(os.path.join(BASE_DIR, f'data/{save}.csv'))

    return anns_info

