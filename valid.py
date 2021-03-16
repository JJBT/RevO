import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import sys
from settings import BASE_DIR
from callbacks import LoadCheckpointCallback

logger = logging.getLogger(__name__)


def run_validation(cfg):
    from trainer import Trainer
    trainer = Trainer(cfg)
    ckpt_dir = os.path.join(BASE_DIR, cfg.path, 'checkpoints')
    trainer.register_callback(LoadCheckpointCallback(
        directory=ckpt_dir,
        filename=cfg.ckpt
    ))
    trainer._before_run_callbacks()
    metrics = trainer.evaluate()
    logger.info(f'Validation {os.path.join(ckpt_dir, cfg.ckpt)}')
    for k, v in metrics.items():
        logger.info(f'{k} : {v:.4f}')

    logger.info('Done')


@hydra.main(config_path='conf', config_name='config_valid')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    trainer_cfg_filename = os.path.join(BASE_DIR, cfg.path, 'cfg', 'config.yaml')
    trainer_cfg = OmegaConf.load(trainer_cfg_filename)
    merged_cfg = OmegaConf.merge(trainer_cfg, cfg)
    print(OmegaConf.to_yaml(merged_cfg))
    # run_validation(merged_cfg)


if __name__ == '__main__':
    run()
