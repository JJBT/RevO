import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
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

    logger.info(f'Validation {os.path.join(ckpt_dir, cfg.ckpt)}')
    for name, dataloader_dict in trainer.val_dataloader_dict.items():
        logger.info(f'Dataset: {name}')

        cats = dataloader_dict['dataloader'].dataset.cats
        for cat_id in cats:
            metrics_report = ''
            metrics_report += f'id: {cat_id} name: {cats[cat_id]["name"]}'.ljust(25)

            dataloader = get_single_cat_dataloader(dataloader_dict['dataloader'], cat_id=cat_id)
            metrics = trainer.evaluate(dataloader=dataloader, metrics=trainer.metrics)
            metrics_report += '|  '
            for k, v in metrics.items():
                metrics_report += f'{k}: {v:.3f}  |  '

            logger.info(metrics_report)

    logger.info('Done')


@hydra.main(config_path='conf', config_name='config_valid')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    trainer_cfg_filename = os.path.join(BASE_DIR, cfg.path, 'cfg', 'config.yaml')
    trainer_cfg = OmegaConf.load(trainer_cfg_filename)
    merged_cfg = OmegaConf.merge(trainer_cfg, cfg)

    run_validation(merged_cfg)


if __name__ == '__main__':
    run()
