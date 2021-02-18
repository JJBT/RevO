from pycocotools.coco import COCO
import numpy as np
import pandas as pd
from os.path import join, isdir
from os import mkdir, makedirs
import sys
import os
import json
import cv2

from settings import COCO_TRAIN_PATH, COCO_KPS_TRAIN_PATH


def crop_support(image, bbox, keypoints, context=16):
    image_h, image_w = image.shape[:2]

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

    crop_x1 = max(x1 - context, 0)
    crop_y1 = max(y1 - context, 0)
    crop_x2 = min(x2 + context + 1, image_w)
    crop_y2 = min(y2 + context + 1, image_h)

    new_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    new_x1 = context if cro

    return new_image, None, None


def main(coco_path, ann_path):
    coco = COCO(ann_path)

    anns_info_df = pd.read_csv('anns_info.csv')
    ids_to_keep = anns_info_df[(anns_info_df['num_keypoints'] == 17) & (anns_info_df['bbox_y_scale'] >= 0.9)]['id'].values

    for ann in coco.loadAnns(ids_to_keep):
        bbox = ann['bbox']
        print(bbox)
        keypoints = ann['keypoints']

        image_meta = coco.loadImgs(ann['image_id'])[0]
        image = cv2.imread(os.path.join(coco_path, image_meta['file_name']))

        support_image, support_bbox, support_kps = crop_support(image, bbox, keypoints)

        cv2.imshow('test', support_image)
        cv2.waitKey(0)
        break


if __name__ == '__main__':
    main(coco_path=COCO_TRAIN_PATH, ann_path=COCO_KPS_TRAIN_PATH)

