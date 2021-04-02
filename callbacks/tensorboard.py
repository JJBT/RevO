import os
import numpy as np
from functools import reduce
from callbacks.callback import Callback
from torch.utils.tensorboard import SummaryWriter
from utils.vis_utils import draw_batch, image_grid


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
        num_images = 12
        num_full_batches, num_remained_images = divmod(num_images, trainer.cfg.bs)
        dataloder_names = [name for name, dataloader in trainer.val_dataloader_dict.items() if dataloader['draw']]


        is_training = trainer.model.training
        trainer.model.eval()

        all_images = []
        for name in dataloder_names:
            data_iter = iter(trainer.val_dataloader_dict[name]['dataloader'])
            for i in range(num_full_batches + 1):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                input = batch['input']
                target = batch['target']
                target = target.to(trainer.device)
                output = trainer.model(input)
                images = draw_batch(input['q_img'], output, target)

                if i == num_full_batches and num_remained_images:
                    images = images[:num_remained_images]

                all_images.append(images)

        all_images = np.concatenate(all_images)
        all_titles = reduce(lambda x, y: x + [y] * num_images, [name for name in dataloder_names], [])

        figure = image_grid(all_images, all_titles)
        self.writer.add_figure('val_prediction_visualization',
                               figure, trainer.state.step, close=True)

        trainer.model.train(is_training)

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
