from pytorchtrainer.metric import Metric, Accuracy, TorchLoss
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
