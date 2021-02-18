from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import json
import sys


def show_annotation_by_id(ann_id, coco, path='D:/datasets/train2017'):
    ann = coco.loadAnns([ann_id])
    image_meta = coco.loadImgs(ann[0]['image_id'])[0]

    im = cv2.imread(os.path.join(path, image_meta['file_name']))
    plt.imshow(im); plt.axis('off')
    plt.gca()
    coco.showAnns(ann)
    plt.show()


def show_image_by_id(image_id, coco, show_ann=False, path='D:/datasets/train2017'):
    image_meta = coco.loadImgs(image_id)[0]
    im = cv2.imread(os.path.join(path, image_meta['file_name']))
    plt.imshow(im)
    plt.axis('off')

    if show_ann:
        annIds = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(annIds)
        plt.gca()
        coco.showAnns(anns)

    plt.show()




