import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import settings
import torch
from torchvision.transforms import Normalize
from PIL import ImageDraw, Image
import numpy as np
from jinja2 import Template
from settings import BASE_DIR
from utils.data import from_yolo_target
# import pdfkit


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

inv_normalize = Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

TEMPLATE_PATH = os.path.join(BASE_DIR, 'utils', 'template.html')


def render_report(images_path, title='', report_name=None):
    if report_name is not None:
        html_path = os.path.join(os.getcwd(), f'{report_name}.html')
        pdf_path = os.path.join(os.getcwd(), f'{report_name}.pdf')
    else:
        html_path = os.path.join(os.getcwd(), 'report.html')
        pdf_path = os.path.join(os.getcwd(), 'report.pdf')

    images = [
        os.path.join(images_path, image) for image in os.listdir(images_path)
    ]

    context = {
        'title': title,
        'images': images
    }

    with open(TEMPLATE_PATH, 'r') as html:
        template = Template(html.read())
        rendered_template = template.render(context)

    with open(html_path, 'w') as result_html:
        result_html.write(rendered_template)

    pdfkit.from_file(html_path, pdf_path)


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


def draw_batch(imgs, outputs, targets, show=False):
    fig = plt.figure(figsize=(14, 14))
    axes = []
    tile_side = math.ceil(math.sqrt(len(imgs)))

    for i, (img, output, target) in enumerate(zip(imgs, outputs, targets)):
        ax = fig.add_subplot(tile_side, tile_side, i + 1)
        fig, ax = draw_sample(img, output, target, fig=fig, ax=ax)
        axes.append(ax)

    if show:
        plt.show()
    return fig, axes


def draw_sample(img, output, target, fig=None, ax=None, show=False):
    img_size = img.shape[1:]
    grid_size = target.shape[:2]

    pred = torch.sigmoid(output)
    pred_bboxes = from_yolo_target(pred, img_size, grid_size)
    target_bboxes = from_yolo_target(target, img_size, grid_size)

    img = inv_normalize(img)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255 / np.max(img)).astype('uint8')

    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    if ax is None:
        ax = fig.add_subplot(111)

    ax.imshow(img)
    for pred_bbox in pred_bboxes:
        pred_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]),
                                      pred_bbox[2], pred_bbox[3],
                                      linewidth=2,
                                      edgecolor='g',
                                      facecolor='none')
        ax.add_patch(pred_rect)

    for target_bbox in target_bboxes:
        target_rect = patches.Rectangle((target_bbox[0], target_bbox[1]),
                                        target_bbox[2], target_bbox[3],
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none')
        ax.add_patch(target_rect)

    if show:
        plt.show()
    return fig, ax


def image_grid(images, titles=None):
    num_images = images.shape[0]
    if titles is None:
        titles = [''] * num_images

    grid_size = math.ceil(math.sqrt(num_images))
    figure = plt.figure(figsize=(14, 14))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1, title=titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])

    return figure
