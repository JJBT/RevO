import torch
from torch import nn
import logging
import time
import os
import dist
from models.backbone import resnet_backbone
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet18
from metrics import Accuracy, TorchLoss
from callbacks import LoadCheckpointCallback, SaveBestCheckpointCallback, \
    SaveCheckpointCallback, ValidationCallback, LogCallback
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from models.prikol import PrikolNet
from datasets.dataset import BBoxDataset


def create_model(cfg):
    device = create_device(cfg)
    input_size = cfg.data.support.input_size
    input_size = (input_size, input_size)
    model = PrikolNet(
        backbone_name=cfg.model.backbone.architecture,
        backbone_pratrained=cfg.model.backbone.pretrained,
        backbone_trainable_layers=cfg.model.backbone.trainable_layers,
        backbone_returned_layers=cfg.model.backbone.returned_layers,
        pool_shape=input_size,
        embd_dim=cfg.model.transformer.embd_dim,
        n_head=cfg.model.transformer.n_head,
        attn_pdrop=cfg.model.transformer.attn_pdrop,
        n_layer=cfg.model.transformer.n_layer,
        out_dim=cfg.model.transformer.out_dim,
        embd_pdrop=cfg.model.transformer.embd_pdrop,
        resid_pdrop=cfg.model.transformer.resid_pdrop,
        device=device
    )
    return model.to(device)


def create_optimizer(cfg, model: torch.nn.Module):
    if cfg.optimizer.type == 'adam':
        optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.optimizer.lr)
        return optimizer
    return None


def create_scheduler(cfg, optimizer: torch.optim.Optimizer):
    # TODO
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


def create_loss(cfg):
    if cfg.loss.type == 'bcewithlogits':
        return nn.BCEWithLogitsLoss()


def create_train_dataloader(cfg):
    batch_size = cfg.train.bs
    params = dict()
    params['q_root'] = cfg.data.query.root
    params['s_root'] = cfg.data.support.root
    params['annFileQuery'] = cfg.data.query.annotation
    params['annFileSupport'] = cfg.data.support.annotation
    params['k_shot'] = cfg.train.k_shot
    input_size = cfg.data.query.input_size
    params['q_img_size'] = (input_size, input_size)
    params['backbone_stride'] = cfg.model.backbone.stride
    q_transform, s_transform = create_augmentations(cfg)
    params['q_transform'] = q_transform
    params['s_transform'] = s_transform

    dataset = BBoxDataset(**params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return dataloader


def create_val_dataloader(cfg):
    # TODO
    return create_train_dataloader(cfg)


def create_metrics(cfg):
    # TODO
    return [Accuracy(), TorchLoss(nn.BCEWithLogitsLoss())]


def create_device(cfg):
    return torch.device(cfg.train.device)


def create_callbacks(cfg, trainer):
    # TODO
    trainer.register_callback(LogCallback(frequency=5))
    trainer.register_callback(ValidationCallback(create_metrics(cfg), frequency=10))
    # trainer.register_callback(SaveCheckpointCallback(frequency=3))
    # trainer.register_callback(SaveBestCheckpointCallback(frequency=2, state_metric_name='last_validation_accuracy'))
    pass


def create_augmentations(cfg):
    q_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2()
    ], bbox_params=albu.BboxParams(format='coco', label_fields=['bboxes_cats']))
    s_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2()
    ], bbox_params=albu.BboxParams(format='coco', label_fields=['bboxes_cats']))
    return q_transform, s_transform
