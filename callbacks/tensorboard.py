import os
from callbacks.callback import Callback
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback(Callback):
    def __init__(self, frequency, add_weights_and_grads=False):
        super().__init__(frequency=frequency, before=True, after=True)
        self.log_dir = os.getcwd()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.add_weights_and_grads = add_weights_and_grads
        self.validation_callback_in_trainer = False

    def before_run(self, trainer):
        self.validation_callback_in_trainer = 'ValidationCallback' in trainer.callbacks
        description = trainer.cfg.description
        if description:
            self.writer.add_text('description', description)

    def after_run(self, trainer):
        self.writer.close()

    def draw_prediction(self, trainer):
        from PIL import Image
        img = Image.open('/home/vladimir/Diploma/img/checkpoint-10000/img0.png')
        import numpy as np
        img = np.array(img)
        self.writer.add_image(f'gs_{trainer.state.step}', img, trainer.state.step, dataformats='HWC')

    def add_validation_metrics(self, trainer):
        for metric in trainer.metrics:
            metric_name = metric.__class__.__name__.lower()
            metric_computed = trainer.state.get(f'last_validation_{metric_name}')
            self.writer.add_scalar(f'vld/{metric_name}', metric_computed, trainer.state.step)

    def __call__(self, trainer):
        self.writer.add_scalar('trn/loss', trainer.state.last_train_loss, trainer.state.step)
        self.writer.add_scalar('lr', trainer.optimizer.param_groups[0]['lr'], trainer.state.step)

        if self.add_weights_and_grads:
            for name, param in trainer.model.named_parameters():
                if 'bn' not in name:
                    self.writer.add_histogram(name, param, trainer.state.step)
                    if param.requires_grad:
                        self.writer.add_histogram(name + '_grad', param.grad, trainer.state.step)
