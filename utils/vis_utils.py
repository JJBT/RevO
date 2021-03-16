import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import settings
import torch
from torchvision.transforms import Normalize
from PIL import ImageDraw, Image
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def show_annotation_by_id(ann_id, coco, path=None):
    if path is None:
        path = settings.COCO_TRAIN_PATH

    ann = coco.loadAnns([ann_id])
    image_meta = coco.loadImgs(ann[0]['image_id'])[0]

    im = cv2.imread(os.path.join(path, image_meta['file_name']))
    plt.imshow(im); plt.axis('off')
    plt.gca()
    coco.showAnns(ann)
    plt.show()


def show_image_by_id(image_id, coco, show_ann=False, path=None):
    if path is None:
        path = settings.COCO_TRAIN_PATH

    image_meta = coco.loadImgs(image_id)[0]
    im = cv2.imread(os.path.join(path, image_meta['file_name']))
    plt.imshow(im)
    plt.axis('off')

    if show_ann:
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        plt.gca()
        coco.showAnns(anns)

    plt.show()


def vis_bboxes(img, bboxes):
    if not isinstance(bboxes[0], list) and not isinstance(bboxes[0], tuple):
        bboxes = [bboxes]

    fig, ax = plt.subplots()

    ax.imshow(img)

    for bbox in bboxes:
        x, y, w, h = bbox  # [x_l, y_t, w, h] -> [x_l, y_b, w, h] (matplotlib считает началом координат левый нижний угол)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def draw(img, output, target, step):
    pred = torch.nonzero(output > 0, as_tuple=False).tolist()
    gt = torch.nonzero(target > 0, as_tuple=False).tolist()
    inv_normalize = Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    img = inv_normalize(img)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255 / np.max(img)).astype('uint8')

    points_true = []
    points_pred = []
    r = 2

    for idx in pred:
        y, x = divmod(idx[0], 10)
        y *= 32
        x *= 32
        y += 16
        x += 16
        p = list()
        p.append((x-r, y-r))
        p.append((x+r, y+r))
        points_pred.append(p)

    for idx in gt:
        y, x = divmod(idx[0], 10)
        y *= 32
        x *= 32
        y += 16
        x += 16
        p = list()
        p.append((x - r, y - r))
        p.append((x + r, y + r))
        points_true.append(p)

    img_pil = Image.fromarray(img)
    img_d = ImageDraw.Draw(img_pil)
    print((target > 0).sum())
    print((output > 0).sum())
    for p in points_true:
        img_d.ellipse(p, fill=(255, 0, 0))
    for p in points_pred:
        img_d.ellipse(p, fill=(0, 255, 0))

    img_pil.save(os.path.join(os.getcwd(), f'output{step}.png'))

