import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
from utils.vis_utils import draw, render_report
from settings import BASE_DIR
from callbacks import LoadCheckpointCallback


logger = logging.getLogger(__name__)


def run_prediction(cfg):
    from trainer import Trainer
    trainer = Trainer(cfg)
    ckpt_dir = os.path.join(BASE_DIR, cfg.path, 'checkpoints')
    trainer.register_callback(LoadCheckpointCallback(
        directory=ckpt_dir,
        filename=cfg.ckpt
    ))
    trainer._before_run_callbacks()
    trainer.model.eval()
    dataloader = trainer.train_dataloader_dict if cfg.dataloader == 'train' else trainer.val_dataloader_dict
    dataloader = dataloader['megapixel_mnist_train_val']['dataloader']
    os.makedirs(os.path.join(os.getcwd(), 'output', os.path.splitext(cfg.ckpt)[0]), exist_ok=True)
    for i, batch in enumerate(dataloader):
        print('before run')
        input_tensor = batch['input']
        target_tensor = batch['target']
        target_tensor = target_tensor.to(trainer.device)
        print('inputs unpacked')
        outputs = trainer.model(input_tensor)
        print('oututs gained')
        res_img = draw(input_tensor['q_img'][0], outputs[0], target_tensor[0])
        print('imgs drawed')
        res_img.save(os.path.join(os.getcwd(), 'output', os.path.splitext(cfg.ckpt)[0], f'img{i}.png'))
        print('imgs saved')
        del outputs
        del input_tensor
        del target_tensor
        logger.info(i)
        if i == 14:
            break
    #render_report(os.path.join(os.getcwd(), 'output', os.path.splitext(cfg.ckpt)[0]))


@hydra.main(config_path='conf', config_name='config_draw')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    trainer_cfg_filename = os.path.join(BASE_DIR, cfg.path, 'cfg', 'config.yaml')
    trainer_cfg = OmegaConf.load(trainer_cfg_filename)
    merged_cfg = OmegaConf.merge(trainer_cfg, cfg)

    run_prediction(merged_cfg)


if __name__ == '__main__':
    run()
