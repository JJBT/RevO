import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import settings


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
