import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging


logger = logging.getLogger(__name__)


def run_train(cfg):
    from trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run_train()


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    run_train(cfg)


if __name__ == '__main__':
    run()
