import signal
import torch
from factory import create_scheduler, create_callbacks, create_model, create_loss, create_optimizer, \
    create_train_dataloader, create_val_dataloader, create_device, create_metrics
import os
from callbacks import Callback, StopAtStep
import logging
from collections import OrderedDict


logger = logging.getLogger(__name__)


class State:
    def __init__(self):
        self.step = 0
        self.last_train_loss = None

        self.validation_metrics = dict()

    def get(self, attribute_name: str):
        return getattr(self, attribute_name)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def state_dict(self):
        return self.__dict__

    def add_attribute(self, name, value):
        if not hasattr(self, name):
            setattr(self, name, value)

    def add_validation_metric(self, name, value):
        if name not in self.validation_metrics:
            self.validation_metrics[name] = []

        self.validation_metrics[name].append(value)

    def reset(self):
        self.step = 0
        self.last_train_loss = None

    def update(self, loss=None):
        self.step += 1
        if loss is not None:
            self.last_train_loss = loss.item()

    def log_train(self):
        logger.info(f'Step - {self.step} loss - {self.last_train_loss:.3f}')

    def log_validation(self):
        msg = f'Validation  '
        for attr in self.__dict__:
            if attr.startswith('last_') and attr != 'last_train_loss':
                msg += f'{attr} - {getattr(self, attr):.4f} '

        logger.info(msg)


class Trainer:
    def __init__(self, cfg):
        signal.signal(signal.SIGINT, self._soft_exit)

        self.train_dataloader = create_train_dataloader(cfg)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = create_val_dataloader(cfg)
        self.state = State()
        self.loss = create_loss(cfg)
        self.model = create_model(cfg)
        self.optimizer = create_optimizer(cfg, self.model)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        self.n_steps = cfg.n_steps
        self.stop_condition = StopAtStep(last_step=self.n_steps)
        self.callbacks = OrderedDict()
        create_callbacks(cfg, self)
        self.metrics = create_metrics(cfg)
        self.cfg = cfg
        self.device = create_device(cfg)
        self.stop_validation = False

    def get_train_batch(self):
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = next(self.train_iter)

        return batch

    def run_step(self, batch):
        # accumulation gradient
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        input_tensor = batch['input']
        target_tensor = batch['target']
        target_tensor = target_tensor.to(self.device)

        outputs = self.model(input_tensor)

        loss = self.loss(outputs, target_tensor)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.detach()

    def run_train(self, n_steps=None):
        if n_steps is not None:
            self.stop_condition = StopAtStep(last_step=n_steps)

        self.state.reset()
        self.model.train()

        self._before_run_callbacks()

        while not self.stop_condition(self.state):
            batch = self.get_train_batch()
            loss = self.run_step(batch)
            self.state.update(loss)

            self._run_callbacks()

        self._after_run_callbacks()
        logger.info('Done')

    def evaluate(self, dataloader=None, metrics=None):
        if dataloader is None:
            dataloader = self.val_dataloader
        if metrics is None:
            metrics = self.metrics

        previous_training_flag = self.model.training

        self.model.eval()
        for metric in metrics:
            metric.reset()

        with torch.no_grad():
            for batch in dataloader:
                input_tensor = batch['input']
                target_tensor = batch['target']
                target_tensor = target_tensor.to(self.device)
                outputs = self.model(input_tensor)

                for metric in metrics:
                    metric.step(y=target_tensor, y_pred=outputs)

                if self.stop_validation:
                    break

        metrics_computed = {metric.name: metric.compute() for metric in metrics}
        self.model.train(previous_training_flag)

        return metrics_computed

    def register_callback(self, callback: Callback):
        callback.set_trainer(self)
        callback_name = callback.__class__.__name__
        self.callbacks[callback_name] = callback

    def _soft_exit(self, sig, frame):
        logger.info('Soft exit... Currently running steps will be finished')
        self.stop_condition = lambda state: True
        self.stop_validation = True

    def _before_run_callbacks(self):
        for name, callback in self.callbacks.items():
            callback.before_run(self)

    def _after_run_callbacks(self):
        for name, callback in self.callbacks.items():
            callback.after_run(self)

    def _run_callbacks(self):
        for name, callback in self.callbacks.items():
            freq = callback.frequency
            if freq != 0 and self.state.step % freq == 0:
                callback(self)
