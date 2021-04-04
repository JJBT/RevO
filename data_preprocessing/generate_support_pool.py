from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import os
import json
import cv2
from copy import copy
from settings import COCO_TRAIN_PATH, COCO_KPS_TRAIN_PATH


def crop_support(image, bbox, keypoints, context=16):
    image_h, image_w = image.shape[:2]
    bbox_w, bbox_h = bbox[2], bbox[3]

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

    crop_x1 = max(x1 - context, 0)
    crop_y1 = max(y1 - context, 0)
    crop_x2 = min(x2 + context + 1, image_w)
    crop_y2 = min(y2 + context + 1, image_h)

    crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_image_w, crop_image_h = crop_x2 - crop_x1, crop_y2 - crop_y1

    new_x1 = context if crop_x1 else x1
    new_y1 = context if crop_y1 else y1
    new_x2 = new_x1 + bbox_w
    new_y2 = new_x1 + bbox_h

    new_bbox = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]

    new_keypoints = copy(keypoints)
    x_delta = x1 - new_x1
    y_delta = y1 - new_y1
    new_keypoints[::3] = list(map(lambda x: x - x_delta, keypoints[::3]))
    new_keypoints[1::3] = list(map(lambda y: y - y_delta, keypoints[1::3]))

    if crop_image_w < crop_image_h:
        new_image = np.zeros((crop_image_h, crop_image_h, 3), dtype=np.uint8)
        shift = (crop_image_h - crop_image_w) // 2
        new_image[:, shift:shift + crop_image_w, :] = crop_image

        new_bbox[0], new_bbox[2] = new_bbox[0] + shift, new_bbox[2] + shift
        new_keypoints[::3] = list(map(lambda x: x + shift, new_keypoints[::3]))

    else:
        new_image = np.zeros((crop_image_w, crop_image_w, 3), dtype=np.uint8)
        shift = (crop_image_w - crop_image_h) // 2
        new_image[shift:shift + crop_image_h, :, :] = crop_image

        new_bbox[1], new_bbox[3] = new_bbox[1] + shift, new_bbox[3] + shift
        new_keypoints[1::3] = list(map(lambda y: y + shift, new_keypoints[1::3]))

    scale = 320 / new_image.shape[0]
    new_image = cv2.resize(src=new_image, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)
    new_bbox = list(map(lambda coord: coord * scale, new_bbox))
    new_keypoints[::3] = list(map(lambda x: x * scale, new_keypoints[::3]))
    new_keypoints[1::3] = list(map(lambda y: y * scale, new_keypoints[1::3]))

    return new_image, new_bbox, new_keypoints


def gen_support_pool(coco_path, ann_path, support_path):
    support_dir = os.path.join(support_path, 'support_pool')
    if not os.path.exists(support_dir):
        os.makedirs(support_dir)

    coco = COCO(ann_path)

    anns_info_df = pd.read_csv('anns_info.csv')
    ids_to_keep = anns_info_df[(anns_info_df['num_keypoints'] == 17) & (anns_info_df['bbox_y_scale'] >= 0.9)]['id'].values

    support_images = []
    support_anns = []
    for i, ann in enumerate(coco.loadAnns(ids_to_keep)):
        bbox = ann['bbox']
        keypoints = ann['keypoints']

        image_meta = coco.loadImgs(ann['image_id'])[0]
        image = cv2.imread(os.path.join(coco_path, image_meta['file_name']))
        support_image, support_bbox, support_kps = crop_support(image, bbox, keypoints)

        support_image_meta = {}
        support_image_meta['file_name'] = str(i).rjust(5, '0') + '.jpg'
        support_image_meta['height'] = support_image.shape[0]
        support_image_meta['weight'] = support_image.shape[1]
        support_image_meta['id'] = i
        support_images.append(support_image_meta)

        support_ann = {}
        support_ann['num_keypoints'] = ann['num_keypoints']
        support_ann['keypoints'] = support_kps
        support_ann['image_id'] = support_image_meta['id']
        support_ann['bbox'] = support_bbox
        support_ann['category_id'] = ann['category_id']
        support_ann['id'] = i
        support_anns.append(support_ann)

        cv2.imwrite(os.path.join(support_path, 'support_pool', support_image_meta['file_name']), support_image)

    with open(ann_path, 'r') as load_file:
        dataset = json.load(load_file)
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_categories = dataset['categories']

    support_coco = {
        'info': save_info,
        'licenses': save_licenses,
        'images': support_images,
        'annotations': support_anns,
        'categories': save_categories
    }

    with open(os.path.join(support_path, 'support_keypoints_17.json'), 'w') as f:
        json.dump(support_coco, f)


if __name__ == '__main__':
    SUPPORT_PATH = './support_17'
    if not os.path.exists(SUPPORT_PATH):
        os.makedirs(SUPPORT_PATH)

    gen_support_pool(coco_path=COCO_TRAIN_PATH, ann_path=COCO_KPS_TRAIN_PATH, support_path=SUPPORT_PATH)
