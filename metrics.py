from pytorchtrainer.metric import Metric, Accuracy
from pytorchtrainer.metric import TorchLoss as PttTorchLoss
import torch


class AveragePrecision(Metric):
    def __init__(self):
        super().__init__('average precision', default_value=0)
        pass

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        pass

    def compute(self):
        pass

    def reset(self):
        pass


class TorchLoss(PttTorchLoss):
    def __init__(self, loss_function: torch.nn.modules.loss, prediction_transform=lambda x: x):
        super().__init__(loss_function)
        self.prediction_transform = prediction_transform

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        y_pred = self.prediction_transform(y_pred)
        loss = self.loss_function(y_pred, y)
        self._loss_sum += loss.item()
        self._total += 1
        return loss
