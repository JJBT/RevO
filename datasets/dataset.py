import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class CustomDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            annFile: str,
            transform=None,
            target_transform=None,
            transforms=None,
    ):
        super(VisionDataset, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx: int):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)
