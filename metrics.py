import torch
from utils.utils import loss_to_dict
from collections import defaultdict
from utils.precision_recall import average_precision_compute
from utils.data import from_yolo_target_torch
from torchvision.ops import box_iou
from utils


class Metric:
    def __init__(self, name: str, default_value=None, target_transform=None, prediction_transform=None):
        self.name = name.replace(' ', '_')
        self.default_value = default_value
        self.target_transform = target_transform if target_transform else lambda x: x
        self.prediction_transform = prediction_transform if prediction_transform else lambda x: x

    def prepare(self, y: torch.Tensor, y_pred: torch.Tensor):
        y = self.target_transform(y)
        y_pred = self.prediction_transform(y_pred)
        if y.size() != y_pred.size():
            raise TypeError("y and y_pred should have the same shape")

        if isinstance(y, torch.Tensor):
            y = y.detach()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach()

        return y, y_pred

    def step(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class APAccumulator:
    def __init__(self, thr_overlap):
        self.thr_overlap = thr_overlap
        self.labels = []
        self.confs = []

    def step(self, mask, pred_bboxes):
        labels = mask.sum(dim=1).bool()
        confs = pred_bboxes[..., 0]
        self.labels.append(labels)
        self.confs.append(confs)

    def compute(self):
        labels = torch.cat(self.labels)
        confs = torch.cat(self.confs)
        return average_precision_compute(preds=confs, target=labels).item()


class AveragePrecision(Metric):
    def __init__(self, target_transform=None, prediction_transform=None, thr_confidence=0):
        super().__init__('ap', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.thr_confidence = thr_confidence
        self.accumulators = {
            'ap': APAccumulator(thr_overlap=0.5)
        }

    @staticmethod
    def compute_IoU_mask(y: torch.Tensor, y_pred: torch.Tensor, overlap_threshold):
        IoU = box_iou(y, y_pred)
        for i in range(y.shape[0]):
            maxj = IoU[i, :].argmax()
            IoU[i, :maxj] = 0
            IoU[i, (maxj + 1):] = 0

        return IoU >= overlap_threshold

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        for i in range(y.shape[0]):
            target_bboxes = from_yolo_target_torch(y[i], [320, 320], [10, 10])
            pred_bboxes = from_yolo_target_torch(y_pred[i], [320, 320], [10, 10])
            for name, accumulator in self.accumulators.items():
                mask = self.compute_IoU_mask(target_bboxes[..., 1:], pred_bboxes[..., 1:], accumulator.thr_overlap)
                accumulator.step(mask, pred_bboxes)

    def compute(self):
        result = dict()
        for name, accumulator in self.accumulators.items():
            result[name] = accumulator.compute()

        return result

    def reset(self):
        for name, accumulator in self.accumulators.items():
            accumulator.labels.clear()
            accumulator.confs.clear()


class Recall(Metric):
    def __init__(self, target_transform=None, prediction_transform=None):
        super().__init__('recall', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self._true_positives = 0
        self._total_positives = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # Just for this case.
        # TODO
        y, y_pred = self.prepare(y, y_pred)

        true_positives = (y * y_pred).sum().item()
        total_positives = y.sum().item()

        self._true_positives += true_positives
        self._total_positives += total_positives

    def compute(self):
        numerator = self._true_positives
        denominator = self._total_positives
        if denominator == 0:
            result = 0.
        else:
            result = numerator / denominator

        return result

    def reset(self):
        self._true_positives = 0
        self._total_positives = 0


class Precision(Metric):
    def __init__(self, target_transform=None, prediction_transform=None):
        super().__init__('precision', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self._true_positives = 0
        self._false_postitives = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        # Just for this case.
        # TODO
        y, y_pred = self.prepare(y, y_pred)

        true_positives = (y * y_pred).sum().item()
        false_positives = ((1 - y) * y_pred).sum().item()

        self._true_positives += true_positives
        self._false_postitives += false_positives

    def compute(self):
        numerator = self._true_positives
        denominator = self._true_positives + self._false_postitives
        if denominator == 0:
            result = 0.
        else:
            result = numerator / denominator

        return result

    def reset(self):
        self._true_positives = 0
        self._false_postitives = 0


class TorchLoss(Metric):
    def __init__(self, loss_function: torch.nn.modules.loss, target_transform=None, prediction_transform=None):
        super().__init__('loss', default_value=float('inf'), target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.loss_function = loss_function
        self._loss_sum_dict = defaultdict(lambda: 0)
        self._total = 0

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        y, y_pred = self.prepare(y, y_pred)

        loss_dict = self.loss_function(y_pred, y)
        loss_dict = loss_to_dict(loss_dict)

        for loss_name, loss_value in loss_dict.items():
            self._loss_sum_dict[loss_name] += loss_value.item()

        self._total += 1

    def compute(self):
        if self._total == 0:
            result = dict.fromkeys(self._loss_sum_dict.keys(), 0)
        else:
            result = dict()
            for loss_name, loss_value in self._loss_sum_dict.items():
                result[loss_name] = loss_value / self._total

        return result

    def reset(self):
        self._loss_sum_dict = defaultdict(lambda: 0)
        self._total = 0
