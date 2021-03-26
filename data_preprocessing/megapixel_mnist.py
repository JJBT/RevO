"""Make an artificial large image dataset using MNIST digits"""

import argparse
import json
from os import path
from torchvision.datasets import MNIST
import numpy as np
from skimage.transform import rescale
import cv2
from data_utils import save_anns


class MegapixelMNIST:
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
                bbox = int(p[0]), int(p[1]), int(p[0] + 28), int(p[1] + 28)
                self._coco_targets.append(
                    {
                        'bbox': bbox,
                        'category_id': int(y)
                    }
                )
            return self._coco_targets

    def __init__(self, N=5000, W=1500, H=1500, train=True, seed=42):
        # Load the images
        mnist = MNIST(root='mnist', train=train, download=False)
        x = mnist.data
        y = mnist.targets
        x = x.numpy()
        y = y.numpy()
        x = x.astype(np.float32) / 255.

        # Save the needed variables to generate high and low res samples
        self._W, self._H = W, H
        self._images = x

        # Generate the dataset
        np.random.seed(seed + int(train))
        self._nums, self._targets, self._n_patches = self._get_numbers(N, y)
        self._pos = self._get_positions(N, W, H, self._n_patches)

    def _get_numbers(self, N, y):
        n_patches = np.random.randint(low=1, high=7, size=N)
        nums = []
        targets = []
        all_idxs = np.arange(len(y))
        for i in range(N):
            idxs = np.random.choice(all_idxs, size=n_patches[i])
            nums.append(idxs)
            targets.append(y[idxs])

        return np.array(nums), np.array(targets), n_patches

    def _get_positions(self, N, W, H, n_patches):
        def overlap(positions, pos):
            if len(positions) == 0:
                return False
            distances = np.abs(
                np.asarray(positions) - np.asarray(pos)[np.newaxis]
            )
            axis_overlap = distances < 28
            return np.logical_and(axis_overlap[:, 0], axis_overlap[:, 1]).any()

        positions = []
        for i in range(N):
            position = []
            for j in range(n_patches[i]):
                while True:
                    pos = np.round(np.random.rand(2) * [H - 28, W - 28]).astype(int)
                    if not overlap(position, pos):
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


def transform_to_coco_format(dataset, root):
    images = []
    annotations = []
    cats = {
        0: {'id': 0, 'name': 'zero'},
        1: {'id': 1, 'name': 'one'},
        2: {'id': 2, 'name': 'two'},
        3: {'id': 3, 'name': 'three'},
        4: {'id': 4, 'name': 'four'},
        5: {'id': 5, 'name': 'five'},
        6: {'id': 6, 'name': 'six'},
        7: {'id': 7, 'name': 'seven'},
        8: {'id': 8, 'name': 'eight'},
        9: {'id': 9, 'name': 'nine'}
    }
    anns_counter = 0
    for i, (img, anns) in enumerate(dataset):
        filename = f'{i:05d}.png'
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

    annotaion = {
        'images': images,
        'annotaions': annotations,
        'categories': cats
    }
    return annotaion


def save_image(img, filename_to_save):
    img = img * 255
    img = img.astype(np.uint8)
    cv2.imwrite(filename_to_save, img)


def main():
    n_train = 5000
    n_test = 1000
    width = 320
    height = 320

    training = MegapixelMNIST(
        N=n_train,
        train=True,
        W=width,
        H=height
    )

    # Write the test set
    test = MegapixelMNIST(
        N=n_test,
        train=False,
        W=width,
        H=height
    )
    # train_annotation = transform_to_coco_format(training, )
    test_annotations = transform_to_coco_format(test, 'mnist/megapixel_mnist/test')
    save_anns(test_annotations, 'mnist/megapixel_mnist/annotations/test.json')


if __name__ == "__main__":
    main()