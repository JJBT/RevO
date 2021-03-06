import torch
import hydra
import omegaconf
from omegaconf import DictConfig
import os
import factory
import logging
from trainer import Trainer

logger = logging.getLogger(__name__)


def run_train(cfg):
    trainer = Trainer(cfg)
    trainer.run_train()


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    run_train(cfg)


if __name__ == '__main__':
    run()
