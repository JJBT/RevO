"""Make an artificial large image dataset using EMNIST characters"""

import argparse
import json
import os
from os import path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import Omniglot, EMNIST
import cv2
import sys
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
sys.path.append('.')
from utils.data import save_coco_anns


class RandomResize(DualTransform):
    def __init__(self, h_resize_limit=1., w_resize_limit=1., interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(RandomResize, self).__init__(always_apply, p)
        if isinstance(h_resize_limit, float):
            assert 0. <= h_resize_limit <= 1.
            self.h_resize_limit = 1 - abs(h_resize_limit), 1 + abs(h_resize_limit)

        elif isinstance(h_resize_limit, tuple) or isinstance(h_resize_limit, list):
            assert all(list(map(lambda x: isinstance(x, float), h_resize_limit)))
            assert all(list(map(lambda x: 0. <= x, h_resize_limit)))
            assert h_resize_limit[0] < h_resize_limit[1]
            self.h_resize_limit = h_resize_limit
        else:
            raise ValueError

        if isinstance(w_resize_limit, float):
            assert 0. <= w_resize_limit <= 1.
            self.w_resize_limit = 1 - abs(w_resize_limit), 1 + abs(w_resize_limit)

        elif isinstance(w_resize_limit, tuple) or isinstance(w_resize_limit, list):
            assert all(list(map(lambda x: isinstance(x, float), w_resize_limit)))
            assert all(list(map(lambda x: 0. <= x, w_resize_limit)))
            assert w_resize_limit[0] < w_resize_limit[1]
            self.w_resize_limit = w_resize_limit
        else:
            raise ValueError

        self.interpolation = interpolation

    def get_params(self):
        return {
            'h_scale': np.random.uniform(self.h_resize_limit[0], self.h_resize_limit[1]),
            'w_scale': np.random.uniform(self.w_resize_limit[0], self.w_resize_limit[1])
        }

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w = int(params['h_scale'] * img.shape[0]), int(params['w_scale'] * img.shape[1])
        return F.resize(img, height=h, width=w, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("h_resize_limit", "w_resize_limit", "interpolation")


class MegapixelEMNIST:
    class Sample:
        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = idxs

            self.size = self.dataset.H, self.dataset.W
            self.n_channels = 3 if self.dataset.is_colourful else 1

        def _get_slice(self, x_l, y_t, w, h):
            return (
                slice(max(0, y_t), max(0, y_t + h)),
                slice(max(0, x_l), max(0, x_l + w)),
                slice(None)
            )

        def _locate_pathces(self, idxs):
            def valid(bbox):
                x, y, w, h = bbox

                if x < 0 or x + w > self.dataset.W:
                    return False
                if y < 0 or y + h > self.dataset.H:
                    return False
                return True

            def overlap(bboxes, bbox):
                # bboxes in coco format [x_l, y_t, w, h]
                if len(bboxes) == 0:
                    return False

                coords = np.asarray(bboxes)[:, :2]
                coord = np.asarray(bbox)[np.newaxis, :2]
                distances = (coords - coord)

                sizes = np.asarray(bboxes)[:, 2:]
                size = np.asarray(bbox)[np.newaxis, 2:]
                limits = np.where(distances < 0, sizes, size)

                axis_overlap = abs(distances) < limits
                return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

            patches = []
            bboxes = []
            bbox_cats = []

            for idx in idxs:
                patch = self.dataset.x[idx]
                bbox = [0, 0, 28, 28]
                bbox_cat = [int(self.dataset.y[idx])]
                transformed = self.dataset.transform(image=patch, bboxes=[bbox], bbox_cats=[bbox_cat])
                patch, bbox = transformed['image'], transformed['bboxes'][0]

                i = 1
                while i < 5000:
                    i += 1
                    x_l, y_t = np.round(np.random.rand(2) * [self.dataset.W - bbox[1], self.dataset.H - bbox[2]])
                    bbox = tuple(map(int, (x_l, y_t, bbox[2], bbox[3])))
                    if valid(bbox) and not overlap(bboxes, bbox):
                        break
                patches.append(patch)
                bboxes.append(bbox)
                bbox_cats.append(bbox_cat)

                if i == 5000:
                    patches.pop(), bboxes.pop(), bbox_cats.pop()

            return patches, bboxes, bbox_cats

        def get_sample(self):
            def coco_target(bboxes, bbox_cats):
                target = []
                for bbox, bbox_cat in zip(bboxes, bbox_cats):
                    target.append(
                        {
                            'bbox': bbox,
                            'category_id': bbox_cat[0]
                        }
                    )
                return target

            image = np.zeros(self.size + (self.n_channels,), dtype=np.uint8)
            patches, bboxes, bbox_cats = self._locate_pathces(idxs=self.idxs)
            for patch, bbox in zip(patches, bboxes):
                if self.dataset.is_colourful:
                    colour = np.random.randint(1, 256, size=(3,))
                else:
                    colour = 255

                image[self._get_slice(*bbox)] = \
                    colour * np.stack((patch,) * self.n_channels, axis=2)

            target = coco_target(bboxes, bbox_cats)
            return image, target

    def __init__(self, root, N=5000, W=1500, H=1500, train=True, cat_ids=None, is_colourful=False, affine=False, seed=42):
        download = not os.path.exists(root)
        self.train = train
        os.makedirs(root, exist_ok=True)
        emnist = EMNIST(root=root, download=download, split='balanced', transform=lambda x: np.array(x).T)

        if cat_ids is None:
            cat_ids = []

        data = list(map(list, zip(*[elem for elem in emnist])))
        x = np.array(data[0])
        y = np.array(data[1])
        x = x.astype(np.float32) / 255.
        self.x = x
        self.y = y
        self.cat_ids = cat_ids
        # self.all_idxs = np.arange(len(self.y))[np.logical_or.reduce([self.y == cat_id for cat_id in self.cat_ids])]

        self.cats_dict = [{'id': idx, 'name': name} for idx, name in enumerate(emnist.classes)]

        self.N = N
        self._cur_n = 0
        self.W, self.H = W, H

        self.is_colourful = is_colourful
        if affine:
            self.transform = albu.Compose([
                albu.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
                RandomResize(h_resize_limit=[0.7, 1.5], w_resize_limit=[0.7, 1.5], p=1.)
            ], bbox_params=albu.BboxParams(format='coco', label_fields=['bbox_cats']))
        else:
            self.transform = albu.Compose([])

        np.random.seed(seed + int(train) + len(cat_ids))

    def _get_sample(self):
        s = 10 if self.train else 0
        cat_ids = self.cat_ids + list(np.random.choice(np.arange(s, len(self.cats_dict)), len(self.cat_ids)))
        all_idxs = np.arange(len(self.y))[np.logical_or.reduce([self.y == cat_id for cat_id in cat_ids])]

        n_patches = np.random.randint(low=1, high=5)
        idxs = np.random.choice(all_idxs, size=n_patches)
        sample = self.Sample(self, idxs).get_sample()

        return sample

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._cur_n < self.__len__():
            self._cur_n += 1
            sample = self._get_sample()
            x, y = (sample[0] / 255.).astype(np.float32), sample[1]
            return x, y

        raise StopIteration()


def transform_to_coco_format(dataset, root, phase=''):
    images = []
    annotations = []
    cats = [dataset.cats_dict[cat_id] for cat_id in dataset.cat_ids]

    anns_counter = 0
    for i, (img, anns) in tqdm(enumerate(dataset), total=len(dataset)):
        filename = phase + f'{i:05d}.png'
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

        for ann in anns:
            if ann['category_id'] in dataset.cat_ids:
                ann_dict = {
                    'id': anns_counter,
                    'bbox': ann['bbox'],
                    'image_id': image_id,
                    'category_id': ann['category_id']
                }
                anns_counter += 1
                annotations.append(ann_dict)

    annotaion = {
        'images': images,
        'annotations': annotations,
        'categories': cats
    }
    return annotaion


def save_image(img, filename_to_save):
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(filename_to_save, img)


def main(omniglot_path, megapixel_omniglot_path):
    is_colourful = True
    affine = True
    n_train = 1000
    n_val = 150
    n_train_val = 150
    train_cat_ids = [0, 2, 4, 5, 7, 8, 9]
    novel_cat_ids = [1, 3, 6]
    width = 112
    height = 112
    
    train = MegapixelEMNIST(
        root=omniglot_path,
        N=n_train,
        train=True,
        cat_ids=train_cat_ids,
        W=width,
        H=height,
        is_colourful=is_colourful,
        affine=affine
    )
    val_train_cats = MegapixelEMNIST(
        root=omniglot_path,
        N=n_train_val,
        train=True,
        cat_ids=train_cat_ids,
        W=width,
        H=height,
        is_colourful=is_colourful,
        affine=affine
    )

    val_novel_cats = MegapixelEMNIST(
        root=omniglot_path,
        N=n_val,
        train=False,
        cat_ids=novel_cat_ids,
        W=width,
        H=height,
        is_colourful=is_colourful,
        affine=affine
    )

    if os.path.exists(megapixel_omniglot_path):
        shutil.rmtree(megapixel_omniglot_path)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'val_novel_cats'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'val_train_cats'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_omniglot_path, 'annotations'), exist_ok=True)
    train_annotations = transform_to_coco_format(train, os.path.join(megapixel_omniglot_path, 'train'), phase='train')
    val_annotations = transform_to_coco_format(val_novel_cats, os.path.join(megapixel_omniglot_path, 'val_novel_cats'), phase='val_novel_cats')
    train_val_annotations = transform_to_coco_format(val_train_cats, os.path.join(megapixel_omniglot_path, 'val_train_cats'), phase='val_train_cats')
    save_coco_anns(train_annotations, os.path.join(megapixel_omniglot_path, 'annotations/train.json'))
    save_coco_anns(val_annotations, os.path.join(megapixel_omniglot_path, 'annotations/val_novel_cats.json'))
    save_coco_anns(train_val_annotations, os.path.join(megapixel_omniglot_path, 'annotations/val_train_cats.json'))


if __name__ == "__main__":
    omniglot_path = 'D:/datasets/emnist/'
    megapixel_omniglot_path = 'D:/datasets/megapixel_emnist_balanced/'
    main(omniglot_path, megapixel_omniglot_path)
