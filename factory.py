import torch
from utils.utils import object_from_dict
from torch.utils.data import DataLoader
import albumentations as albu
from models.prikol import PrikolNet
from datasets.dataset import ObjectPresenceDataset, object_presence_collate_fn
from utils.pred_transforms import prediction_transforms_dict


def create_backbone(cfg):
    backbone = object_from_dict(cfg)
    return backbone


def create_model(cfg):
    device = create_device(cfg)
    input_size = cfg.data.input_size
    backbone = create_backbone(cfg.model.backbone)
    model = object_from_dict(cfg.model, backbone=backbone, pool_shape=input_size, device=device)
    return model.to(device)


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
    train_dataloaders = dict()
    for dataset_cfg in cfg.data.train_dataset:
        dataloader_dict = create_dataloader(dataset_cfg)
        train_dataloaders[dataloader_dict['name']] = dataloader_dict
    return train_dataloaders


def create_val_dataloader(cfg):
    val_dataloaders = dict()
    for dataset_cfg in cfg.data.validation_dataset:
        dataloader_dict = create_dataloader(dataset_cfg)
        val_dataloaders[dataloader_dict['name']] = dataloader_dict
    return val_dataloaders


def create_dataloader(cfg):
    batch_size = cfg.bs
    params = dict()
    params['q_root'] = cfg.query.root
    params['s_root'] = cfg.support.root
    params['q_ann_filename'] = cfg.query.annotations
    params['s_ann_filename'] = cfg.support.annotations
    params['k_shot'] = cfg.k_shot
    params['q_img_size'] = cfg.input_size
    params['backbone_stride'] = cfg.backbone_stride
    q_transform = create_augmentations(cfg.transforms)
    s_transform = create_augmentations(cfg.transforms)
    params['q_transform'] = q_transform
    params['s_transform'] = s_transform

    dataset = ObjectPresenceDataset(**params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=cfg.shuffle, collate_fn=object_presence_collate_fn)
    dataloader_dict = {
        'name': cfg.name,
        'dataloader': dataloader,
        'draw': cfg.draw,
    }

    return dataloader_dict


def create_metrics(cfg):
    metrics = []
    for metric in cfg.metrics:
        metric_name = metric.type.split('.')[-1].lower()
        prediction_transform = prediction_transforms_dict.get(metric_name, lambda x: x)
        metric_obj = object_from_dict(metric, prediction_transform=prediction_transform)
        metrics.append(metric_obj)

    return metrics


def create_device(cfg):
    return torch.device(cfg.device)


def create_callbacks(cfg, trainer):
    for hook in cfg.hooks:
        hook_obj = object_from_dict(hook)
        trainer.register_callback(hook_obj)


def create_augmentations(cfg):
    augmentations = []
    for augm in cfg:
        augmentations.append(object_from_dict(augm))

    transform = albu.Compose(augmentations,
                             bbox_params=albu.BboxParams(format='coco', label_fields=['bboxes_cats']))
    return transform
