import signal
import torch
from factory import create_logger, create_loss, create_model, create_optimizer, create_val_dataloader, \
    create_train_dataloader, create_metrics, create_callbacks
import os
import callbacks


logger = create_logger(os.getcwd())


class State:
    def __init__(self):
        self.step = 0
        self.last_train_loss = None

    def set(self, state):
        if not isinstance(state, State):
            raise TypeError("state argument should be of type State.")

        for k, v in state.__dict__.items():
            setattr(self, k, v)

    def reset(self):
        self.step = 0
        self.last_train_loss = None

    def update(self, loss=None):
        self.step += 1
        if loss is not None:
            self.last_train_loss = loss


class Trainer:
    def __init__(self, cfg):
        signal.signal(signal.SIGINT, self.__soft_exit)

        self.train_dataloader = create_train_dataloader(cfg)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = create_val_dataloader(cfg)
        self.state = State()
        self.loss = create_loss(cfg)
        self.model = create_model(cfg)
        self.optimizer = create_optimizer(cfg, self.model)
        self.n_steps = 5
        self.stop_condition = callbacks.StopAtStep(last_step=self.n_steps)
        self.metrics = create_metrics(cfg)
        self.callbacks = []
        create_callbacks(cfg, self)

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
            self.stop_condition = callbacks.StopAtStep(last_step=n_steps)

        self.state.reset()
        self.model.train()

        while not self.stop_condition(self.state):
            batch = self.get_train_batch()
            self.run_step(batch)
            self.__run_callbacks()

    def evaluate(self, dataloader=None, metrics=None):
        if dataloader is None:
            dataloader = self.val_dataloader
        if metrics is None:
            metrics = self.metrics

        previous_training_flag = self.model.training

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in dataloader:
                input_tensor = batch['input']
                target_tensor = batch['target']
                outputs = self.model(input_tensor.float())

                metrics.step(y=outputs, y_pred=target_tensor)

        self.model.train(previous_training_flag)
        return metrics.compute()

    def register_callback(self, callback: callbacks.Callback, frequency=1):
        if frequency < 0:
            raise ValueError("frequency argument should be positive.")
        callback.set_trainer(self)
        self.callbacks.append((frequency, callback))

    def __soft_exit(self, sig, frame):
        logger.info('Currently running steps will be finished')
        self.stop_condition = lambda state: True

    def __run_callbacks(self):
        for frequency, callback in self.callbacks:
            if frequency != 0 and self.state.step % frequency == 0:
                callback(self)
