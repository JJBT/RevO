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


def create_model(cfg: dict):
    # TODO
    return resnet18()


def create_optimizer(cfg: dict, model: torch.nn.Module):
    # TODO
    return torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()))


def create_scheduler(cfg: dict, optimizer: torch.optim.Optimizer):
    # TODO
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


def create_loss(cfg: dict):
    # TODO
    return nn.CrossEntropyLoss()


def create_train_dataloader(cfg: dict):
    # TODO
    dataset = []
    for _ in range(40):
        sample = {'input': np.random.randn(3, 224, 224), 'target': np.random.randint(low=1, high=5)}
        dataset.append(sample)

    return DataLoader(dataset, batch_size=2)


def create_val_dataloader(cfg: dict):
    # TODO
    return create_train_dataloader(cfg)


def create_metrics(cfg):
    # TODO
    return [Accuracy()]


def create_callbacks(cfg, trainer):
    # TODO
    # trainer.register_callback(ValidationCallback(create_metrics(cfg), frequency=2))
    # trainer.register_callback(SaveCheckpointCallback(frequency=3))
    # trainer.register_callback(SaveBestCheckpointCallback(frequency=2, state_metric_name='last_validation_accuracy'))
    trainer.register_callback(LoadCheckpointCallback('outputs/2021-03-06/10-58-42/checkpoints/', 'checkpoint-3.pt'))
