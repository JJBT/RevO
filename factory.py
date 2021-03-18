import torch
from utils.utils import object_from_dict
from torch.utils.data import DataLoader
from callbacks import SaveBestCheckpointCallback, \
    SaveCheckpointCallback, ValidationCallback, LogCallback, TensorBoardCallback
import albumentations as albu
from models.prikol import PrikolNet
from datasets.dataset import BBoxDataset
from pred_transforms import prediction_transforms_dict


def create_model(cfg):
    device = create_device(cfg)
    input_size = cfg.data.input_size
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
    optimizer = object_from_dict(cfg.optimizer, params=filter(lambda x: x.requires_grad, model.parameters()))
    return optimizer


def create_scheduler(cfg, optimizer: torch.optim.Optimizer):
    scheduler = object_from_dict(cfg.scheduler, optimizer=optimizer)
    return scheduler


def create_loss(cfg):
    loss = object_from_dict(cfg.loss)
    return loss


def create_train_dataloader(cfg):
    batch_size = cfg.train.bs
    params = dict()
    params['q_root'] = cfg.data.train.query.root
    params['s_root'] = cfg.data.train.support.root
    params['annFileQuery'] = cfg.data.train.query.annotation
    params['annFileSupport'] = cfg.data.train.support.annotation
    params['k_shot'] = cfg.train.k_shot
    input_size = cfg.data.train.query.input_size
    params['q_img_size'] = (input_size, input_size)
    params['backbone_stride'] = cfg.model.backbone.stride
    q_transform = create_augmentations(cfg.data.train.query)
    s_transform = create_augmentations(cfg.data.train.support)
    params['q_transform'] = q_transform
    params['s_transform'] = s_transform

    dataset = BBoxDataset(**params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return dataloader


def create_val_dataloader(cfg):
    batch_size = cfg.train.bs
    params = dict()
    params['q_root'] = cfg.data.validation.query.root
    params['s_root'] = cfg.data.validation.support.root
    params['annFileQuery'] = cfg.data.validation.query.annotation
    params['annFileSupport'] = cfg.data.validation.support.annotation
    params['k_shot'] = cfg.train.k_shot
    input_size = cfg.data.validation.query.input_size
    params['q_img_size'] = (input_size, input_size)
    params['backbone_stride'] = cfg.model.backbone.stride
    q_transform = create_augmentations(cfg.data.validation.query)
    s_transform = create_augmentations(cfg.data.validation.support)
    params['q_transform'] = q_transform
    params['s_transform'] = s_transform

    dataset = BBoxDataset(**params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return dataloader


def create_metrics(cfg):
    metrics = []
    for metric in cfg.metrics:
        metric_dict = dict()
        metric_dict['type'] = metric
        metric_name = metric.split('.')[-1].lower()
        metric_dict['prediction_transform'] = prediction_transforms_dict.get(metric_name, lambda x: x)
        metric_obj = object_from_dict(metric_dict)
        metrics.append(metric_obj)

    return metrics


def create_device(cfg):
    return torch.device(cfg.train.device)


def create_callbacks(cfg, trainer):
    metrics = []
    hooks_cfg = cfg.train.hooks
    if 'log' in hooks_cfg:
        trainer.register_callback(LogCallback(frequency=hooks_cfg.log.frequency))

    if 'validation' in hooks_cfg:
        metrics = create_metrics(hooks_cfg.validation)
        trainer.register_callback(ValidationCallback(metrics, frequency=hooks_cfg.validation.frequency))

    if 'tensorboard' in hooks_cfg:
        trainer.register_callback(TensorBoardCallback(frequency=hooks_cfg.tensorboard.frequency))

    if 'save_checkpoint' in hooks_cfg:
        trainer.register_callback(SaveCheckpointCallback(frequency=hooks_cfg.save_checkpoint.frequency))

    if 'save_best_checkpoint' in hooks_cfg:
        trainer.register_callback(SaveBestCheckpointCallback(frequency=hooks_cfg.save_best_checkpoint.frequency,
                                                             num=hooks_cfg.save_best_checkpoint.num,
                                                             state_metric_name=hooks_cfg.save_best_checkpoint.state_metric_name))
    return metrics


def create_augmentations(cfg):
    augmentations = []
    for augm in cfg.transform:
        augm_dict = dict()
        augm_dict['type'] = augm
        if augm == 'albumentations.Resize':
            augm_dict['height'] = cfg.input_size
            augm_dict['width'] = cfg.input_size

        augmentations.append(object_from_dict(augm_dict))

    transform = albu.Compose(augmentations,
                             bbox_params=albu.BboxParams(format=cfg.bbox_format, label_fields=['bboxes_cats']))
    return transform
