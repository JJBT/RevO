import cv2
import matplotlib.pyplot as plt
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




