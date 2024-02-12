import os
from callbacks.callback import Callback
import torch
from utils.utils import get_state_dict, load_state_dict
from settings import BASE_DIR
import logging
import warnings


logger = logging.getLogger(__name__)


class SaveCheckpointCallback(Callback):
    def __init__(self, frequency=0, num=0, n_steps=None):
        if not frequency ^ num:
            raise ValueError('Should specify either frequency or num')

        if num and n_steps:
            frequency = n_steps / num
        elif not frequency:
            raise ValueError('If num specified n_steps should be specified too')

        super().__init__(frequency=frequency, before=False, after=True)
        cwd = os.getcwd()
        self.savedir = os.path.join(cwd, 'checkpoints')
        os.makedirs(self.savedir, exist_ok=True)
        self.ckpt_filename = 'checkpoint-{}.pt'

    def __call__(self, trainer):
        self._save_checkpoint(trainer, self.ckpt_filename.format(trainer.state.step))

    def _save_checkpoint(self, trainer, filename):
        torch.save({
            'model_state_dict': get_state_dict(trainer.model),
            # 'optimizer_state_dict': get_state_dict(trainer.optimizer),
            # 'scheduler_state_dict': get_state_dict(trainer.scheduler),
            'trainer_state': get_state_dict(trainer.state),
            'model_class': str(trainer.model.__class__),
            # 'optimizer_class': str(trainer.optimizer.__class__),
            # 'scheduler_class': str(trainer.scheduler.__class__)
        }, os.path.join(self.savedir, filename))


class LoadCheckpointCallback(Callback):
    def __init__(self, directory: str, filename=None):
        super().__init__(frequency=0, before=True, after=False)
        self.directory = os.path.join(BASE_DIR, directory)

        self.filename = filename if filename is not None else self._search_checkpoint()
        self.filename_to_load = os.path.join(self.directory, self.filename)

    def __call__(self, trainer):
        self._load_checkpoint(trainer)
        logger.info(f'Checkpoint {self.filename_to_load} loaded')

    def _load_checkpoint(self, trainer):
        checkpoint = torch.load(self.filename_to_load, map_location=trainer.accelerator.device)

        # checks
        if checkpoint['model_class'] != str(trainer.model.__class__):
            warnings.warn(
                f'Models do not match: {checkpoint["model_class"]} and {trainer.model.__class__}', RuntimeWarning
            )

        load_state_dict(trainer.model, checkpoint['model_state_dict'])
        # trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.state.load_state_dict(checkpoint['trainer_state'])

    def _search_checkpoint(self):
        filelist = os.listdir(self.directory)
        for file in filelist:
            if '.pt' in file:
                return file


class SaveBestCheckpointCallback(SaveCheckpointCallback):
    def __init__(self, frequency, state_metric_name: str, num=0, comparison_function=lambda metric, best: metric < best):
        super().__init__(frequency=frequency, num=num)
        self.state_metric_name = state_metric_name  # last_(train/validation)_{metric}
        self.comparison_function = comparison_function
        self.current_best = None
        self.best_ckpt_filename = 'best-checkpoint-{}.pt'

    def __call__(self, trainer):
        self.state_last_metric = trainer.state.get(self.state_metric_name)
        if self.current_best is None or self.comparison_function(self.state_last_metric, self.current_best):
            self.current_best = self.state_last_metric

            self._save_checkpoint(trainer, self.best_ckpt_filename.format(trainer.state.step))

