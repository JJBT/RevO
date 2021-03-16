from pytorchtrainer.metric import Metric, Accuracy
from pytorchtrainer.metric import TorchLoss as PttTorchLoss
import torch
from typing import cast


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


class Recall(Metric):
    def __init__(self, prediction_transform=lambda x: x):
        super().__init__('recall', default_value=0)
        self._true_positives = 0
        self._total_positives = 0
        self.prediction_transform = prediction_transform

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # Just for this case.
        # TODO
        y_pred = self.prediction_transform(y_pred)
        if y.size() != y_pred.size():
            raise TypeError("y and y_pred should have the same shape")
        true_positives = (y * y_pred).sum().item()
        total_positives = y.sum().item()

        self._true_positives += true_positives
        self._total_positives += total_positives

    def compute(self):
        result = self._true_positives / self._total_positives

        return result

    def reset(self):
        self._true_positives = 0
        self._total_positives = 0


class Precision(Metric):
    def __init__(self, prediction_transform=lambda x: x):
        super().__init__('precision', default_value=0)
        self._true_positives = 0
        self._false_postitives = 0
        self.prediction_transform = prediction_transform

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # Just for this case.
        # TODO
        y_pred = self.prediction_transform(y_pred)
        if y.size() != y_pred.size():
            raise TypeError("y and y_pred should have the same shape")
        true_positives = (y * y_pred).sum().item()
        false_positives = ((1 - y) * y_pred).sum().item()

        self._true_positives += true_positives
        self._false_postitives += false_positives

    def compute(self):
        result = self._true_positives / (self._true_positives + self._false_postitives)

        return result

    def reset(self):
        self._true_positives = 0
        self._false_postitives = 0


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
