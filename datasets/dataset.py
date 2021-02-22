import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import albumentations as albu
from vis_utils import vis_bboxes
import settings
from matplotlib import pyplot as plt
K_SHOT = 3


class BBoxDataset(Dataset):
    def __init__(
            self,
            q_root: str,
            s_root: str,
            annFileQuery: str,
            annFileSupport: str,
            q_transform=None,
            s_transform=None
    ):
        super(Dataset, self).__init__()

        self.q_root = q_root
        self.s_root = s_root
        self.q_transform = q_transform
        self.s_transform = s_transform

        from pycocotools.coco import COCO
        self.q_coco = COCO(annFileQuery)
        self.s_coco = self.q_coco
        # self.s_coco = COCO(annFileSupport)
        self.q_ids = sorted(get_coco_img_ids(self.q_coco))
        self.s_ids = sorted(get_coco_img_ids(self.s_coco))

    def __getitem__(self, idx: int):
        sample = {}

        q_coco = self.q_coco
        q_img_id = self.q_ids[idx]
        q_ann_ids = q_coco.getAnnIds(imgIds=q_img_id)
        q_anns = q_coco.loadAnns(q_ann_ids)
        q_target = list(map(lambda ann: ann['bbox'], q_anns))

        q_path = q_coco.loadImgs(q_img_id)[0]['file_name']
        q_img = self._imread(os.path.join(self.q_root, q_path), cv2.COLOR_BGR2RGB)

        s_coco = self.s_coco
        s_img_ids = [self.s_ids[idx]
                     for idx in np.random.randint(len(self.s_ids), size=(K_SHOT,))]
        s_anns = []
        s_targets = []
        s_imgs = []
        for s_img_id in s_img_ids:
            s_ann_id = s_coco.getAnnIds(imgIds=s_img_id)
            s_ann = s_coco.loadAnns(s_ann_id)

            s_target = list(map(lambda ann: ann['bbox'], s_ann))

            s_anns.append(s_ann)
            s_targets.append(s_target)

            s_path = q_coco.loadImgs(s_img_id)[0]['file_name']
            s_img = self._imread(os.path.join(self.s_root, s_path), cv2.COLOR_BGR2RGB)
            # print(f'support shape: {s_img.shape}, image id: {s_img_id}')
            s_imgs.append(s_img)

        if self.q_transform:
            q_transformed = self.q_transform(image=q_img, bboxes=q_target)
            q_img = q_transformed['image']
            q_target = q_transformed['bboxes']

        if self.s_transform:
            s_transformed = [
                self.s_transform(image=s_img, bboxes=s_target)
                for s_img, s_target in zip(s_imgs, s_targets)
            ]
            s_imgs = [transformed['image'] for transformed in s_transformed]
            s_targets = [transformed['bboxes'] for transformed in s_transformed]

        sample['q_img'] = q_img
        sample['q_target'] = q_target
        sample['s_imgs'] = s_imgs
        sample['s_targets'] = s_targets

        return sample

    def __len__(self) -> int:
        return len(self.q_ids)

    def _imread(self, filename, flags=None):
        img = cv2.imread(filename, flags)

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img


def bbox_collate_fn(batch):
    q_img_batched = default_collate([sample['q_img'] for sample in batch])
    # [item for sublist in t for item in sublist]
    s_imgs_batched = default_collate([sample['s_imgs'] for sample in batch])

    q_target = [sample['q_target'] for sample in batch]
    q_target_max_len = max(map(len, q_target))
    q_target_padded = list(map(lambda target:
                               target + list(np.zeros((q_target_max_len - len(target), 4))), q_target))
    q_target_batched = default_collate(q_target_padded)

    sample_batched = {}
    sample_batched['q_img'] = q_img_batched
    sample_batched['q_target'] = q_target_batched
    sample_batched['s_imgs'] = s_imgs_batched
    return sample_batched


def get_coco_img_ids(coco):
    img_ids = set()
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_ids.add(ann['image_id'])

    return list(img_ids)


if __name__ == '__main__':
    dataset = BBoxDataset(q_root=settings.COCO_TRAIN_PATH,
                          s_root=settings.COCO_TRAIN_PATH,
                          annFileQuery=settings.COCO_KPS_TRAIN_PATH,
                          annFileSupport=settings.COCO_KPS_TRAIN_PATH,
                          q_transform=albu.Resize(320, 320),
                          s_transform=albu.Resize(320, 320))

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=bbox_collate_fn)

    for i, batch in enumerate(dataloader):
        print(batch)
        break

