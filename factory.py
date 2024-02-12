import torch
from utils.utils import object_from_dict
from torch.utils.data import DataLoader, SubsetRandomSampler
import albumentations as albu
import numpy as np


def create_backbone(cfg):
    return object_from_dict(cfg)


def create_model(cfg):
    input_size = cfg.data.input_size
    backbone = create_backbone(cfg.model.backbone)
    model = object_from_dict(cfg.model, backbone=backbone, pool_shape=input_size)
    return model


def create_optimizer(cfg, model: torch.nn.Module):
    optimizer = object_from_dict(cfg.optimizer, params=filter(lambda x: x.requires_grad, model.parameters()))
    return optimizer


def create_scheduler(cfg, optimizer: torch.optim.Optimizer):
    scheduler = object_from_dict(cfg.scheduler, optimizer=optimizer)
    return scheduler


def create_loss(cfg):
    loss = object_from_dict(cfg.loss)
    return loss


def create_train_dataloader(cfg):
    train_dataloaders = {}
    for dataset_cfg in cfg.data.train_dataset:
        dataset = create_dataset(dataset_cfg)
        dataloader_dict = create_dataloader(dataset_cfg, dataset)
        train_dataloaders[dataloader_dict['name']] = dataloader_dict
    return train_dataloaders


def create_val_dataloader(cfg):
    val_dataloaders = {}
    for dataset_cfg in cfg.data.validation_dataset:
        dataset = create_dataset(dataset_cfg)
        dataloader_dict = create_dataloader(dataset_cfg, dataset)
        val_dataloaders[dataloader_dict['name']] = dataloader_dict
    return val_dataloaders


def create_dataset(cfg):
    params = {
        'type': cfg.type,
        'q_root': cfg.query.root,
        's_root': cfg.support.root,
        'q_ann_filename': cfg.query.annotations,
        's_ann_filename': cfg.support.annotations,
        'k_shot': cfg.k_shot,
        'q_img_size': cfg.input_size,
        'backbone_stride': cfg.backbone_stride,
    }

    q_transform = create_augmentations(cfg.transforms)
    s_transform = create_augmentations(cfg.transforms)
    params['q_transform'] = q_transform
    params['s_transform'] = s_transform

    return object_from_dict(params)


def create_dataloader(cfg, dataset):
    batch_size = cfg.bs
    shuffle = cfg.shuffle

    if dataset_length := cfg.len:
        if shuffle:
            idx = np.random.choice(len(dataset), dataset_length, replace=False)
            shuffle = False
        else:
            idx = np.arange(dataset_length)

        sampler = SubsetRandomSampler(indices=idx)
    else:
        sampler = None

    collate_fn = object_from_dict(cfg.collate_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            sampler=sampler, collate_fn=collate_fn)
    return {
        'name': cfg.name,
        'dataloader': dataloader,
        'draw': cfg.draw,
    }


def create_metrics(cfg):
    metrics = []
    for metric in cfg.metrics:
        metric_obj = object_from_dict(metric)
        metrics.append(metric_obj)

    return metrics


def create_device(cfg):
    return torch.device(cfg.device)


def create_callbacks(cfg, trainer):
    for hook in cfg.hooks:
        hook_obj = object_from_dict(hook)
        trainer.register_callback(hook_obj)


def create_augmentations(cfg):
    augmentations = [object_from_dict(augm) for augm in cfg]
    return albu.Compose(
        augmentations,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['bboxes_cats']
        ),
    )
