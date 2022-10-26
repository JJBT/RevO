import os
import numpy as np
from functools import reduce

import torch

from callbacks.callback import Callback
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils.vis_utils import draw_batch, image_grid
from omegaconf import OmegaConf


class TensorBoardCallback(Callback):
    def __init__(self, frequency, add_weights=False, add_grads=False):
        super().__init__(frequency=frequency, before=True, after=True)
        self.log_dir = os.getcwd()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.add_weights = add_weights
        self.add_grads = add_grads

    def before_run(self, trainer):
        cfg = OmegaConf.to_yaml(trainer.cfg)
        cfg = cfg.replace('\n', '  \n')
        self.writer.add_text('cfg', cfg)
        if description := trainer.cfg.description:
            self.writer.add_text('description', description)

    def after_run(self, trainer):
        self.writer.close()

    def draw_prediction(self, trainer):
        num_images = 3
        num_full_batches, num_remained_images = divmod(num_images, trainer.cfg.bs)
        dataloder_names = [name for name, dataloader in trainer.val_dataloader_dict.items() if dataloader['draw']]
        if not dataloder_names:
            return

        previous_training_flag = trainer.model.training
        trainer.model.eval()

        for name in dataloder_names:
            all_images = []
            all_outputs = []
            all_targets = []
            data_iter = iter(trainer.val_dataloader_dict[name]['dataloader'])
            with torch.no_grad():
                for i in range(num_full_batches + 1):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break

                    inputs = batch['input']
                    targets = batch['target']
                    outputs = trainer.model(inputs)
                    images = inputs['q_img']

                    if i == num_full_batches and num_remained_images:
                        targets = targets[:num_remained_images]
                        outputs = outputs[:num_remained_images]
                        images = images[:num_remained_images]

                    all_outputs.append(outputs)
                    all_targets.append(targets)
                    all_images.append(images)
                    del batch, inputs, targets, outputs

            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
            all_images = torch.cat(all_images)
            fig, _ = draw_batch(all_images, all_outputs, all_targets)
            self.writer.add_figure(f'{name} visulization',
                                   fig, trainer.state.step, close=True)

        trainer.model.train(previous_training_flag)

    def add_validation_metrics(self, trainer):
        metrics = trainer.state.validation_metrics
        for name, metric in metrics.items():
            self.writer.add_scalar(name, metric, trainer.state.step)

    def add_weights_histogram(self, trainer):
        for name, param in trainer.model.named_parameters():
            if 'bn' not in name:
                self.writer.add_histogram(name, param, trainer.state.step)

    def add_grads_histogram(self, trainer):
        for name, param in trainer.model.named_parameters():
            if 'bn' not in name and param.requires_grad:
                self.writer.add_histogram(f'{name}_grad', param.grad, trainer.state.step)

    def __call__(self, trainer):
        for name, loss in trainer.state.last_train_loss.items():
            self.writer.add_scalar(f'trn/{name}', loss, trainer.state.step)

        self.writer.add_scalar('lr', trainer.optimizer.param_groups[0]['lr'], trainer.state.step)

        if self.add_weights:
            self.add_weights_histogram(trainer)

        if self.add_grads:
            self.add_grads_histogram(trainer)
