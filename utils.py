import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from PIL import ImageDraw, Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def data2device(data, device):
    return data.to(device)


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


def get_state_dict(model):
    if model is None:
        return None
    else:
        return unwrap_model(model).state_dict()


def load_state_dict(model, state_dict):
    if model is None:
        return None
    else:
        return model.load_state_dict(state_dict)


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
