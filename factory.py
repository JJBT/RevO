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
from metrics import Accuracy
from callbacks import LoadCheckpointCallback, SaveBestCheckpointCallback, SaveCheckpointCallback, ValidationCallback
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from models.prikol import PrikolNet
from datasets.dataset import BBoxDataset


def create_model(cfg):
    model = PrikolNet(
        backbone_name=cfg.model.backbone.architecture,
        backbone_pratrained=cfg.model.backbone.pretrained,
        backbone_trainable_layers=cfg.model.backbone.trainable_layers,
        backbone_returned_layers=cfg.model.backbone.returned_layers,
        pool_shape=cfg.data.support.input_size,
        embd_dim=cfg.model.transformer.embd_dim,
        n_head=cfg.model.transformer.n_head,
        attn_pdrop=cfg.model.transformer.attn_pdrop,
        n_layer=cfg.model.transformer.n_layer,
        out_dim=cfg.model.transformer.out_dim,
        embd_pdrop=cfg.model.transformer.embd_pdrop,
        resid_pdrop=cfg.model.transformer.resid_pdrop
    )
    return model


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
    dataset = BB
    return


def create_val_dataloader(cfg: dict):
    # TODO
    return create_train_dataloader(cfg)


def create_metrics(cfg):
    # TODO
    return []


def create_callbacks(cfg, trainer):
    # TODO
    # trainer.register_callback(ValidationCallback(create_metrics(cfg), frequency=2))
    # trainer.register_callback(SaveCheckpointCallback(frequency=3))
    # trainer.register_callback(SaveBestCheckpointCallback(frequency=2, state_metric_name='last_validation_accuracy'))
    trainer.register_callback(LoadCheckpointCallback('outputs/2021-03-06/10-58-42/checkpoints/', 'checkpoint-3.pt'))


def create_augmentations(cfg):
    q_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2()
    ], bbox_params=albu.BboxParams(format='coco', label_fields=['bboxes_cats'])),
    s_transform = albu.Compose([
        albu.Resize(320, 320),
        albu.Normalize(),
        ToTensorV2()
    ], bbox_params=albu.BboxParams(format='coco', label_fields=['bboxes_cats']))
    return q_transform, s_transform
