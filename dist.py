import torch
import torch.distributed as dist
import os


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    initialized = is_dist_available_and_initialized()
    if initialized:
        rank = dist.get_rank()
    else:
        rank = 0

    return rank


def get_world_size():
    initialized = is_dist_available_and_initialized()
    if initialized:
        world_size = dist.get_world_size()
    else:
        world_size = 1

    return world_size
