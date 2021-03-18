import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate


def get_coco_img_ids(coco):
    img_ids = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_ids.add(ann['image_id'])

    return list(img_ids)


def get_object_presence_map(bboxes, shape, stride):
    w, h = shape
    map_w, map_h = w // stride, h // stride
    cell_w, cell_h = w // map_w, h // map_h

    presence = [0] * (map_w * map_h)
    for i, bbox in enumerate(bboxes):
        img_x, img_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        cell_x, cell_y = int(img_x / cell_w), int(img_y / cell_h)
        presence[map_w * cell_y + cell_x] = 1.

    return presence


class ObjectPresenceDataset(Dataset):
    def __init__(
            self,
            q_root,
            s_root,
            annFileQuery,
            annFileSupport,
            k_shot,
            q_img_size,
            backbone_stride,
            q_transform=None,
            s_transform=None
    ):
        super(Dataset, self).__init__()
        self.k_shot = k_shot
        self.q_root = q_root
        self.s_root = s_root
        self.q_img_size = q_img_size
        self.backbone_stride = backbone_stride
        self.q_transform = q_transform
        self.s_transform = s_transform

        from pycocotools.coco import COCO
        self.q_coco = COCO(annFileQuery)
        # self.s_coco = self.q_coco
        self.s_coco = COCO(annFileSupport)
        self.q_ids = sorted(get_coco_img_ids(self.q_coco))
        self.s_ids = sorted(get_coco_img_ids(self.s_coco))

    def __getitem__(self, idx: int):
        """

        Returns:
             sample (dict): data sample
                sample['input'] (dict): input data
                    sample['input']['q_img'] (np.ndarray or torch.Tensor): query image
                    sample['input']['s_imgs'] (List[np.ndarray] or List[torch.Tensor]): K_SHOT support images
                    sample['input']['s_bboxes'] (List[List[float]]): bbox coordinates for support images
                sample['target'] (List[float]): target object presence map (vector actually)
        """
        q_coco = self.q_coco
        q_img_id = self.q_ids[idx]
        q_ann_ids = q_coco.getAnnIds(imgIds=q_img_id)
        q_anns = q_coco.loadAnns(q_ann_ids)
        q_bbox = list(map(lambda ann: ann['bbox'], q_anns))
        q_bbox_cats = list(map(lambda ann: ann['category_id'], q_anns))

        q_path = q_coco.loadImgs(q_img_id)[0]['file_name']
        q_img = self._imread(os.path.join(self.q_root, q_path), cv2.COLOR_BGR2RGB)

        s_coco = self.s_coco
        s_img_ids = [self.s_ids[idx]
                     for idx in np.random.randint(len(self.s_ids), size=(self.k_shot,))]
        s_anns = []
        s_bboxes = []
        s_bboxes_cats = []
        s_imgs = []
        for s_img_id in s_img_ids:
            s_ann_id = s_coco.getAnnIds(imgIds=s_img_id)
            s_ann = s_coco.loadAnns(s_ann_id)

            s_bbox = list(map(lambda ann: ann['bbox'], s_ann))
            s_bbox_cats = list(map(lambda ann: ann['category_id'], s_ann))

            s_anns.append(s_ann)
            s_bboxes.append(s_bbox)
            s_bboxes_cats.append(s_bbox_cats)

            s_path = q_coco.loadImgs(s_img_id)[0]['file_name']
            s_img = self._imread(os.path.join(self.s_root, s_path), cv2.COLOR_BGR2RGB)
            s_imgs.append(s_img)

        if self.q_transform:
            q_transformed = self.q_transform(image=q_img, bboxes=q_bbox, bboxes_cats=q_bbox_cats)
            q_img = q_transformed['image']
            q_bbox = list(map(list, q_transformed['bboxes']))

        if self.s_transform:
            s_transformed = [
                self.s_transform(image=s_img, bboxes=s_target, bboxes_cats=s_target_cats)
                for s_img, s_target, s_target_cats in zip(s_imgs, s_bboxes, s_bboxes_cats)
            ]
            s_imgs = [transformed['image'] for transformed in s_transformed]
            s_bboxes = [list(map(list, transformed['bboxes'])) for transformed in s_transformed]

        target = get_object_presence_map(q_bbox, self.q_img_size, self.backbone_stride)
        sample = {'input': {}, 'target': []}
        sample['input']['q_img'] = q_img
        sample['input']['s_imgs'] = s_imgs
        sample['input']['s_bboxes'] = s_bboxes
        sample['target'] = target
        return sample

    def __len__(self) -> int:
        return len(self.q_ids)

    @staticmethod
    def _imread(filename, flags=None):
        img = cv2.imread(filename, flags)

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img


def object_presence_collate_fn(batch):
    q_img_batched = default_collate([sample['input']['q_img'] for sample in batch])
    target_batched = torch.as_tensor([sample['target'] for sample in batch])

    if type(batch[0]['input']['s_imgs'][0]).__module__ == 'numpy':
        s_imgs_batched = default_collate(np.array([np.array(sample['input']['s_imgs']) for sample in batch]))
    elif type(batch[0]['input']['s_imgs'][0]).__module__ == 'torch':
        s_imgs_batched = torch.stack([torch.stack(sample['input']['s_imgs']) for sample in batch])
    else:
        raise TypeError('Unknown type of support image')

    s_bboxes_batched = [sample['input']['s_bboxes'] for sample in batch]

    sample_batched = {'input': {}, 'target': []}
    sample_batched['input']['q_img'] = q_img_batched
    sample_batched['input']['s_imgs'] = s_imgs_batched
    sample_batched['input']['s_bboxes'] = s_bboxes_batched
    sample_batched['target'] = target_batched

    return sample_batched




