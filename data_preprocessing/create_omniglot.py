"""Make an artificial image dataset using Omniglot characters"""

import argparse
import json
import os
from os import path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import Omniglot
import cv2
import sys
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F

sys.path.append('.')
from utils.data import save_coco_anns


def transform_to_coco_format(dataset, root, phase=''):
    images = []
    annotations = []
    cat_ids = list(range(len(dataset._characters)))
    cats_dict = [{'id': i, 'name': label} for i, label in enumerate(dataset._characters)]
    cats = [cats_dict[cat_id] for cat_id in cat_ids]

    for anns_counter, (i, (img, anns)) in enumerate(tqdm(enumerate(dataset), total=len(dataset))):
        img = np.array(img)
        filename = f'{phase}{i:05d}.png'
        width = int(img.shape[0])
        height = int(img.shape[1])
        image_id = i
        image_dict = {
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height
        }
        images.append(image_dict)

        save_image(img, path.join(root, filename))

        # print(type(ann['bbox'][0]))
        ann_dict = {
            'id': anns_counter,
            'image_id': image_id,
            'category_id': anns
        }
        annotations.append(ann_dict)

    return {'images': images, 'annotations': annotations, 'categories': cats}


def save_image(img, filename_to_save):
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(filename_to_save, img)


def main(omniglot_path, megapixel_omniglot_path):
    train_cat_ids = list(range(964))
    novel_cat_ids = list(range(659))

    train = Omniglot(root=omniglot_path, background=True)
    val_novel_cats = Omniglot(root=omniglot_path, background=False)

    if os.path.exists(megapixel_omniglot_path):
        shutil.rmtree(megapixel_omniglot_path)

    os.makedirs(os.path.join(megapixel_omniglot_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'val_novel_cats'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'annotations'), exist_ok=True)
    train_annotations = transform_to_coco_format(train, os.path.join(megapixel_omniglot_path, 'train'),
                                                 phase='train')
    val_annotations = transform_to_coco_format(val_novel_cats, os.path.join(megapixel_omniglot_path, 'val_novel_cats'),
                                               phase='val_novel_cats')

    save_coco_anns(train_annotations, os.path.join(megapixel_omniglot_path, 'annotations/train.json'))
    save_coco_anns(val_annotations, os.path.join(megapixel_omniglot_path, 'annotations/val_novel_cats.json'))


if __name__ == "__main__":
    omniglot_path = '//data/omniglot/'
    megapixel_omniglot_path = '//data/new_omniglot/'
    main(omniglot_path, megapixel_omniglot_path)
