"""Make an artificial large image dataset using MNIST digits"""

import argparse
import json
import os
from os import path
import shutil
import numpy as np
from torchvision.datasets import MNIST
import cv2
import sys
sys.path.append('.')
from utils.data import save_coco_anns


class MegapixelMNISTGrided:
    """Randomly position several numbers in an image either downsampled or full
    scale and compare the performance of attention sampling.
    Put 5 MNIST images in one large image. Three of them are of the target
    class and the rest are random.
    """

    class Sample:
        def __init__(self, dataset, idxs, positions, targets):
            self._dataset = dataset
            self._idxs = idxs
            self._positions = positions
            self._targets = targets
            self._coco_targets = []

            self._high = None

        def _get_slice(self, pos, s=28, scale=1, offset=(0, 0)):
            pos = (int(pos[0] * scale - offset[0]), int(pos[1] * scale - offset[1]))
            s = int(s)
            return (
                slice(max(0, pos[0]), max(0, pos[0] + s)),
                slice(max(0, pos[1]), max(0, pos[1] + s)),
                0
            )

        def high(self):
            if self._high is None:
                size = self._dataset._H, self._dataset._W
                high = np.zeros(size + (1,), dtype=np.uint8)
                for p, i in zip(self._positions, self._idxs):
                    high[self._get_slice(p)] = \
                        255 * self._dataset._images[i]

                self._high = high
            return self._high

        def create_target(self):
            for p, y in zip(self._positions, self._targets):
                bbox = int(p[1]), int(p[0]), 28, 28
                self._coco_targets.append(
                    {
                        'bbox': bbox,
                        'category_id': int(y)
                    }
                )
            return self._coco_targets

    def __init__(self, root, N=5000, W=1500, H=1500, train=True, cat_ids=None, grid_size=4, seed=42):
        if cat_ids is None:
            cat_ids = []
        # Load the images
        download = not os.path.exists(root)
        os.makedirs(root, exist_ok=True)
        mnist = MNIST(root=root, train=train, download=download)

        x = mnist.data
        y = mnist.targets
        x = x.numpy()
        y = y.numpy()
        x = x.astype(np.float32) / 255.
        self.cat_ids = cat_ids
        self.grid_size = grid_size

        # Save the needed variables to generate high and low res samples
        self._W, self._H = W, H
        self._images = x

        # Generate the dataset
        np.random.seed(seed + int(train))
        self._nums, self._targets, self._n_patches = self._get_numbers(N, y)
        self._pos = self._get_positions(N, W, H, self._n_patches)

    def _get_numbers(self, N, y):
        n_patches = np.random.randint(low=1, high=5, size=N)
        nums = []
        targets = []
        all_idxs = np.arange(len(y))[np.logical_or.reduce([y == cat_id for cat_id in self.cat_ids])]
        for i in range(N):
            idxs = np.random.choice(all_idxs, size=n_patches[i])
            nums.append(idxs)
            targets.append(y[idxs])

        return np.array(nums), np.array(targets), n_patches

    def _get_positions(self, N, W, H, n_patches):
        def valid(pos):
            x, y = pos

            return False if x < 0 or x + 28 > self._W else y >= 0 and y + 28 <= self._H

        def overlap(positions, pos):
            if len(positions) == 0:
                return False
            distances = np.abs(
                np.asarray(positions) - np.asarray(pos)[np.newaxis]
            )
            axis_overlap = distances < 28
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        cell_size = self._W / self.grid_size

        positions = []
        for i in range(N):
            position = []
            for j in range(n_patches[i]):
                while True:
                    cell_idx = np.random.randint(self.grid_size ** 2)
                    cell_y, cell_x = list(map(lambda c: c * cell_size, divmod(cell_idx, self.grid_size)))

                    cell_xc, cell_yc = cell_x + cell_size // 2, cell_y + cell_size // 2

                    x_c, y_c = cell_x + 16, cell_y + 16

                    x_lt, y_lt = cell_x, cell_y
                    x_lt += cell_xc - x_c
                    y_lt += cell_yc - y_c
                    pos = x_lt,  y_lt

                    if valid(pos) and not overlap(position, pos):
                        break
                position.append(pos)
            positions.append(position)

        return np.array(positions)

    def __len__(self):
        return len(self._nums)

    def __getitem__(self, i):
        if len(self) <= i:
            raise IndexError()
        sample = self.Sample(
            self,
            self._nums[i],
            self._pos[i],
            self._targets[i]
        )
        x = sample.high().astype(np.float32) / 255
        y = sample.create_target()
        return x, y


def transform_to_coco_format(dataset, root, phase=''):
    images = []
    annotations = []
    cats = [[{'id': 0, 'name': 'zero'},
             {'id': 1, 'name': 'one'},
             {'id': 2, 'name': 'two'},
             {'id': 3, 'name': 'three'},
             {'id': 4, 'name': 'four'},
             {'id': 5, 'name': 'five'},
             {'id': 6, 'name': 'six'},
             {'id': 7, 'name': 'seven'},
             {'id': 8, 'name': 'eight'},
             {'id': 9, 'name': 'nine'}][cat_id] for cat_id in dataset.cat_ids]

    anns_counter = 0
    for i, (img, anns) in enumerate(dataset):
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

        for ann in anns:
            ann_dict = {
                'id': anns_counter,
                'bbox': ann['bbox'],
                'image_id': image_id,
                'category_id': ann['category_id']
            }
            anns_counter += 1
            annotations.append(ann_dict)

    return {'images': images, 'annotations': annotations, 'categories': cats}


def save_image(img, filename_to_save):
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(filename_to_save, img)


def main(mnist_path, megapixel_mnist_path):
    grid_size = 7
    n_train = 5000
    n_val = 150
    n_train_val = 150
    train_cat_ids = [0, 2, 4, 5, 7, 8, 9]
    novel_cat_ids = [1, 3, 6]
    width = 112
    height = 112

    train = MegapixelMNISTGrided(
        root=mnist_path,
        N=n_train,
        train=True,
        cat_ids=train_cat_ids,
        grid_size=grid_size,
        W=width,
        H=height
    )
    val_train_cats = MegapixelMNISTGrided(
        root=mnist_path,
        N=n_train_val,
        train=True,
        cat_ids=train_cat_ids,
        grid_size=grid_size,
        W=width,
        H=height
    )

    val_novel_cats = MegapixelMNISTGrided(
        root=mnist_path,
        N=n_val,
        train=False,
        cat_ids=novel_cat_ids,
        grid_size=grid_size,
        W=width,
        H=height
    )

    if os.path.exists(megapixel_mnist_path):
        shutil.rmtree(megapixel_mnist_path)
    os.makedirs(os.path.join(megapixel_mnist_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_mnist_path, 'val_novel_cats'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_mnist_path, 'val_train_cats'), exist_ok=True)
    os.makedirs(os.path.join(megapixel_mnist_path, 'annotations'), exist_ok=True)
    train_annotations = transform_to_coco_format(train, os.path.join(megapixel_mnist_path, 'train'), phase='train')
    val_annotations = transform_to_coco_format(val_novel_cats, os.path.join(megapixel_mnist_path, 'val_novel_cats'), phase='val_novel_cats')
    train_val_annotations = transform_to_coco_format(val_train_cats, os.path.join(megapixel_mnist_path, 'val_train_cats'), phase='val_train_cats')
    save_coco_anns(train_annotations, os.path.join(megapixel_mnist_path, 'annotations/train.json'))
    save_coco_anns(val_annotations, os.path.join(megapixel_mnist_path, 'annotations/val_novel_cats.json'))
    save_coco_anns(train_val_annotations, os.path.join(megapixel_mnist_path, 'annotations/val_train_cats.json'))


if __name__ == "__main__":
    mnist_path = ''
    megapixel_mnist_path = ''
    main(mnist_path, megapixel_mnist_path)
