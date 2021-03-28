import os
from callbacks.callback import Callback
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.vis_utils import draw_batch


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
        num_batches = 3
        for name, dataloader_dict in trainer.val_dataloader.items():
            if not dataloader_dict['draw']:
                continue
            dataloader = dataloader_dict['dataloader']
            for i, batch in enumerate(dataloader):
                input = batch['input']
                target = batch['target']
                target = target.to(trainer.device)
                output = trainer.model(input)
                images = draw_batch(input['q_img'], output, target)

                self.writer.add_images(f'{name}_{trainer.state.step}',
                                       images, trainer.state.step, dataformats='NHWC')

                if i > num_batches:
                    break

    def add_validation_metrics(self, trainer):
        metrics = trainer.state.validation_metrics
        for name, metric in metrics.items():
            self.writer.add_scalar(name, metric, trainer.state.step)

    def __call__(self, trainer):
        self.writer.add_scalar('trn/loss', trainer.state.last_train_loss, trainer.state.step)
        self.writer.add_scalar('lr', trainer.optimizer.param_groups[0]['lr'], trainer.state.step)

        if self.add_weights_and_grads:
            for name, param in trainer.model.named_parameters():
                if 'bn' not in name:
                    self.writer.add_histogram(name, param, trainer.state.step)
                    if param.requires_grad:
                        self.writer.add_histogram(name + '_grad', param.grad, trainer.state.step)
