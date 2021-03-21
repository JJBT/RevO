import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import settings
import torch
from torchvision.transforms import Normalize
from PIL import ImageDraw, Image
import numpy as np
from jinja2 import Template
from settings import BASE_DIR
import pdfkit


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
TEMPLATE_PATH = os.path.join(BASE_DIR, 'utils', 'template.html')
REPORT_PATH_HTML = os.path.join(os.getcwd(), 'report.html')
REPORT_PATH_PDF = os.path.join(os.getcwd(), 'report.pdf')


def render_report(images_path, title=''):
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

    with open(REPORT_PATH_HTML, 'w') as result_html:
        result_html.write(rendered_template)

    pdfkit.from_file(REPORT_PATH_HTML, REPORT_PATH_PDF)


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


def draw(img, output, target):
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
    r_gt = 5
    r_pr = 2

    for idx in pred:
        y, x = divmod(idx[0], 10)
        y *= 32
        x *= 32
        y += 16
        x += 16
        p = list()
        p.append((x-r_pr, y-r_pr))
        p.append((x+r_pr, y+r_pr))
        points_pred.append(p)

    for idx in gt:
        y, x = divmod(idx[0], 10)
        y *= 32
        x *= 32
        y += 16
        x += 16
        p = list()
        p.append((x - r_gt, y - r_gt))
        p.append((x + r_gt, y + r_gt))
        points_true.append(p)

    img_pil = Image.fromarray(img)
    img_d = ImageDraw.Draw(img_pil)
    for p in points_true:
        img_d.ellipse(p, outline=(255, 0, 0), width=3)
    for p in points_pred:
        img_d.ellipse(p, outline=(0, 255, 0), width=3)

    return img_pil

