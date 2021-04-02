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
# import pdfkit


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
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


def draw_batch(img, output, target):
    n = img.shape[0]
    images = []
    for i in range(n):
        image = draw(img[i], output[i], target[i])
        image = np.array(image)
        images.append(image)

    return np.stack(images)


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
    r_gt = 8
    r_pr = 5

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
        img_d.ellipse(p, outline=(255, 0, 0), width=4)
    for p in points_pred:
        img_d.ellipse(p, fill=(0, 255, 0), outline=(0, 255, 0), width=3)

    return img_pil


def image_grid(images, titles):
    num_images = images.shape[0]
    grid_size = math.ceil(math.sqrt(num_images))
    figure = plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1, title=titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])

    return figure
