import torch
import torch.distributed as distributed
import os


def is_dist_available_and_initialized():
    if not distributed.is_available():
        return False
    if not distributed.is_initialized():
        return False
    return True


def get_rank():
    initialized = is_dist_available_and_initialized()
    if initialized:
        rank = distributed.get_rank()
    else:
        rank = 0

    return rank


def is_master():
    return get_rank() == 0


def get_world_size():
    initialized = is_dist_available_and_initialized()
    if initialized:
        world_size = distributed.get_world_size()
    else:
        world_size = 1

    return world_size
