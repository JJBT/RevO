import time
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pydoc
from omegaconf.dictconfig import DictConfig


def object_from_dict(d, parent=None, **default_kwargs):
    assert isinstance(d, (dict, DictConfig)) and 'type' in d
    kwargs = d.copy()
    kwargs = dict(kwargs)
    object_type = kwargs.pop('type')

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, (dict, DictConfig)) and 'type' in value:
            value = object_from_dict(value)
            kwargs[key] = value

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def set_determenistic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def data2device(data, device):
    return data.to(device)


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
