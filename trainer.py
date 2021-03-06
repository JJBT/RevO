import signal
import torch
from factory import *
import os
from callbacks.stop_criterion import StopAtStep
from callbacks.callback import Callback

import logging


logger = logging.getLogger(__name__)


class State:
    def __init__(self):
        self.step = 0
        self.last_train_loss = None

    def get(self, attribute_name: str):
        return getattr(self, attribute_name)

    def set(self, state):
        if not isinstance(state, State):
            raise TypeError("state argument should be of type State.")

        for k, v in state.__dict__.items():
            setattr(self, k, v)

    def add_attribute(self, name, value):
        if not hasattr(self, name):
            setattr(self, name, value)

    def reset(self):
        self.step = 0
        self.last_train_loss = None

    def update(self, loss=None):
        self.step += 1
        if loss is not None:
            self.last_train_loss = loss


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
        self.n_steps = cfg.train.n_steps
        self.stop_condition = StopAtStep(last_step=self.n_steps)
        self.metrics = create_metrics(cfg)  # list
        self.callbacks = []
        create_callbacks(cfg, self)
        self.cfg = cfg

    def get_train_batch(self):
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = next(self.train_iter)

        return batch

    def run_step(self, batch):
        # accumulation gradient
        self.optimizer.zero_grad()
        input_tensor = batch['input']
        target_tensor = batch['target']
        outputs = self.model(input_tensor.float())
        loss = self.loss(outputs, target_tensor)
        loss.backward()
        self.optimizer.step()
        self.state.update(loss.detach())

    def run_train(self, n_steps=None):
        if n_steps is not None:
            self.stop_condition = StopAtStep(last_step=n_steps)

        self.state.reset()
        self.model.train()

        self._before_run_callbacks()

        while not self.stop_condition(self.state):
            batch = self.get_train_batch()
            self.run_step(batch)
            self._run_callbacks()

        self._after_run_callbacks()

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
                outputs = self.model(input_tensor.float())

                for metric in metrics:
                    metric.step(y=outputs, y_pred=target_tensor)

        metrics_computed = {metric.name: metric.compute() for metric in metrics}
        self.model.train(previous_training_flag)

        return metrics_computed

    def register_callback(self, callback: Callback):
        callback.set_trainer(self)
        self.callbacks.append(callback)

    def _soft_exit(self, sig, frame):
        logger.info('Soft exit... Currently running steps will be finished')
        self.stop_condition = lambda state: True

    def _before_run_callbacks(self):
        for callback in self.callbacks:
            callback.before_run(self)

    def _after_run_callbacks(self):
        for callback in self.callbacks:
            callback.after_run(self)

    def _run_callbacks(self):
        for callback in self.callbacks:
            freq = callback.frequency
            if freq != 0 and self.state.step % freq == 0:
                callback(self)
