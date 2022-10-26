import torch
import torch.distributed as distributed
import os


def is_dist_available_and_initialized():
    return (
        bool(distributed.is_initialized())
        if distributed.is_available()
        else False
    )


def get_rank():
    return (
        distributed.get_rank()
        if (initialized := is_dist_available_and_initialized())
        else 0
    )


def is_master():
    return get_rank() == 0


def get_world_size():
    return (
        distributed.get_world_size()
        if (initialized := is_dist_available_and_initialized())
        else 1
    )
