import torch
from torchvision.datasets import Omniglot
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np


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


class Patch:
    def __init__(self, root):
        self.dataset = Omniglot(
            root=root,
            background=True,
            transform=lambda x: 255 - np.array(x)
        )
        self.W, self.H = 420, 420
        self.transform = albu.Compose([
                albu.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
                RandomResize(h_resize_limit=[0.7, 1.5], w_resize_limit=[0.7, 1.5], p=1.)
            ], bbox_params=albu.BboxParams(format='coco', label_fields=['bbox_cats']))

        self.cats_dict = [{'id': i, 'name': label} for i, label in enumerate(self.dataset._characters)]
        data = list(map(list, zip(*[elem for elem in self.dataset])))
        x = np.array(data[0])
        y = np.array(data[1])
        x = x.astype(np.float32) / 255.
        self.x = x
        self.y = y

        self.all_idxs = np.arange(len(y))

    def get_sample(self, img):
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

        image = np.zeros((self.H, self.W, 3), dtype=np.uint8)
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

    def _locate_patches(self, img, idxs):
        def valid(bbox):
            x, y, w, h = bbox

            if x < 0 or x + w > self.W:
                return False
            if y < 0 or y + h > self.H:
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
            patch = self.x[idx]
            bbox = [0, 0, 105, 105]
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

    def __call__(self, s_img):
        n_patches = np.random.randint(low=1, high=4)
        result = []
        idxs = np.random.choice(self.all_idxs, size=n_patches)


def preprocess_input(image_list):
    """
    :param image_list: List[PIL.Image]. image_list[0] - query, image_list[1:] - supports
    :return: result (dict): input sample to model
                sampe['q_img'] (torch.Tensor): query image (1xCxHxW)
                sampe['s_imgs'] (torch.Tensor): support set images (1xKxCxHxW)
                sampe['s_bboxes'] (List[List[List[float, ..]]]): bbox coordinates for support
                                                                 set images (conditionally 1xKxVx4, K - length of support set
                                                                                                    V - number of instances per image)
    """
    q_img = image_list[0]
    s_imgs = image_list[1:]
    s_bboxes = [[[0, 0, 105, 105, 1]] for _ in range(3)]

    q_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2()
    ])
    s_transform = albu.Compose([
        albu.PadIfNeeded(420, 420, border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2(),
    ], bbox_params=albu.BboxParams(format='coco'))

    q_img = q_transform(image=q_img)['image']
    s_transformed = [s_transform(image=s_img, bboxes=s_bbox) for s_img, s_bbox in zip(s_imgs, s_bboxes)]
    s_imgs = [transformed['image'] for transformed in s_transformed]
    s_bboxes = [transformed['bboxes'] for transformed in s_transformed]
    s_bboxes = [[s_bbox[0][:-1]] for s_bbox in s_bboxes]

    q_img = torch.unsqueeze(q_img, dim=0)
    s_imgs = torch.stack(s_imgs)
    s_imgs = torch.unsqueeze(s_imgs, dim=0)
    s_bboxes = [[[s_bbox for s_bbox in b_s_bbox] for b_s_bbox in s_bboxes]]
    result = {
        'q_img': q_img,
        's_imgs': s_imgs,
        's_bboxes': s_bboxes
    }
    return result







