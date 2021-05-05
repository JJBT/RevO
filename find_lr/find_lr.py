import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from matplotlib import pyplot as plt
from settings import BASE_DIR

logger = logging.getLogger(__name__)


def log_lr_range_test_history(num_iter, history):
    lr_history, loss_history = history['lr'], history['loss']
    msg = '\n'.join(
        f'Step {i + 1}/{num_iter}\nlr: {lr:.6}, loss: {loss:.6}'
        for i, (lr, loss) in enumerate(zip(lr_history, loss_history))
    )
    logger.info(msg)


class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        return batch_data['input'], batch_data['target']


class CustomValIter(ValDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        return batch_data['input'], batch_data['target']


def net_optimizer(optimizer, model):
    params = model.parameters()
    defaults = optimizer.__getstate__()['defaults']
    net_optimizer = optimizer.__class__(params, **defaults)

    return net_optimizer


def auto_num_iter(data_iter):
    epoch_size = len(data_iter.data_loader)
    num_iter = 4 * epoch_size
    return num_iter


def find_lr(cfg):
    from trainer import Trainer
    trainer = Trainer(cfg)
    custom_train_iter = CustomTrainIter(next(iter(trainer.train_dataloader_dict.values()))['dataloader'])
    custom_val_iter = CustomValIter(next(iter(trainer.val_dataloader_dict.values()))['dataloader'])

    optimizer = net_optimizer(trainer.optimizer, trainer.model)
    if cfg.num_iter == 'auto':
        num_iter = auto_num_iter(custom_train_iter)
    else:
        num_iter = cfg.num_iter

    lr_finder = LRFinder(trainer.model, optimizer, trainer.criterion, device=trainer.device)
    lr_finder.reset()
    if cfg.strategy == 'training_loss':
        lr_finder.range_test(custom_train_iter, end_lr=cfg.end_lr, num_iter=num_iter, step_mode=cfg.step_mode)
    elif cfg.strategy == 'validation_loss':
        lr_finder.range_test(custom_train_iter, val_loader=custom_val_iter,
                             end_lr=cfg.end_lr, num_iter=num_iter, step_mode=cfg.step_mode)
    else:
        raise ValueError(f'expected one of (training_loss, validation_loss), got {cfg.strategy}')

    fig = plt.figure(figsize=(10, 5))
    ax_1 = fig.add_subplot(1, 2, 1)
    result = lr_finder.plot(skip_start=cfg.plot.skip_start, skip_end=cfg.plot.skip_end,
                           log_lr=cfg.plot.log_lr, show_lr=cfg.plot.show_lr, ax=ax_1, suggest_lr=cfg.plot.suggest_lr)
    if isinstance(result, tuple):
        ax_1, suggested_lr = result
    else:
        ax_1 = result
        suggested_lr = 'unspecified'
    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.set_axis_off()
    text = f'Learning rate range test done.\nApproach based on {" ".join(cfg.strategy.split("_"))} used.\n' \
          f'Experiment params:\n' \
          f'  start_lr: {cfg.start_lr}\n' \
          f'  end_lr: {cfg.end_lr}\n' \
          f'  num_iter: {num_iter}\n' \
          f'  step_mode: {cfg.step_mode}\n' \
          f'Experiment result:\n' \
          f'  suggested_lr: {suggested_lr}'
    ax_2.text(0.01, 0.99, text, verticalalignment='top', horizontalalignment='left', fontsize=14)
    plt.savefig('lr_range_test.jpg')
    log_lr_range_test_history(num_iter, lr_finder.history)


@hydra.main(config_path='../conf', config_name='config_find_lr')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    trainer_cfg_filename = os.path.join(BASE_DIR, '../conf', 'config.yaml')
    trainer_cfg = OmegaConf.load(trainer_cfg_filename)
    merged_cfg = OmegaConf.merge(trainer_cfg, cfg)

    find_lr(merged_cfg)


if __name__ == '__main__':
    run()
