import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

logger = logging.getLogger(__name__)


def run_train(cfg):
    logging.info(f'Output directory - {os.getcwd()}')
    from trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run_train()
    logging.info(f'Output directory - {os.getcwd()}')


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    run_train(cfg)


if __name__ == '__main__':
    run()
