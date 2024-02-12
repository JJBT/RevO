import torch
from utils.utils import loss_to_dict
from collections import defaultdict
from utils.precision_recall import average_precision_compute
from utils.data import xywh2xyxy
from torchvision.ops import box_iou, nms
from utils.pre_metrics_transform import transforms_dict


class Metric:
    def __init__(self, name: str, default_value=None, target_transform=None, prediction_transform=None):
        self.name = name.replace(' ', '_')
        self.default_value = default_value
        self.target_transform = target_transform or transforms_dict.get(
            f'{self.name}_target', lambda x: x
        )

        self.prediction_transform = prediction_transform or transforms_dict.get(
            f'{self.name}_prediction', lambda x: x
        )

    def prepare(self, y: torch.Tensor, y_pred: torch.Tensor):
        y = self.target_transform(y)
        y_pred = self.prediction_transform(y_pred)

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
        labels = mask.sum(dim=0).bool()
        confs = pred_bboxes[..., 0]
        self.labels.append(labels)
        self.confs.append(confs)

    def compute(self):
        if len(self.labels) == 0 or len(self.confs) == 0:
            return 0
        labels = torch.cat(self.labels)
        confs = torch.cat(self.confs)
        return average_precision_compute(preds=confs, target=labels).item()


class AveragePrecision(Metric):
    def __init__(self, target_transform=None, prediction_transform=None):
        super().__init__('ap', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.accumulators = {
            'ap': APAccumulator(thr_overlap=0.5)
        }

    @staticmethod
    def get_iou_mask(y: torch.Tensor, y_pred: torch.Tensor, overlap_threshold):
        iou = box_iou(y, y_pred)
        for i in range(y.shape[0]):
            max_idx = iou[i, :].argmax()
            iou[i, :max_idx] = 0
            iou[i, (max_idx + 1):] = 0

        return iou >= overlap_threshold

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        for i in range(y.shape[0]):
            target_bboxes, pred_bboxes = self.prepare(y[i], y_pred[i])

            target_bboxes[..., 1:] = xywh2xyxy(target_bboxes[..., 1:])
            pred_bboxes[..., 1:] = xywh2xyxy(pred_bboxes[..., 1:])
            for name, accumulator in self.accumulators.items():
                mask = self.get_iou_mask(target_bboxes[..., 1:], pred_bboxes[..., 1:], accumulator.thr_overlap)
                accumulator.step(mask, pred_bboxes)

    def compute(self):
        return {
            name: accumulator.compute()
            for name, accumulator in self.accumulators.items()
        }

    def reset(self):
        for name, accumulator in self.accumulators.items():
            accumulator.labels.clear()
            accumulator.confs.clear()


class IoU(Metric):
    def __init__(self, thr=0.5, target_transform=None, prediction_transform=None):
        super().__init__('iou', default_value=0, target_transform=target_transform,
                         prediction_transform=prediction_transform)
        self.total_score = 0
        self.total_samples = 0
        self.thr = thr

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        for i in range(y.shape[0]):
            target_bboxes, pred_bboxes = self.prepare(y[i], y_pred[i])

            target_bboxes[..., 1:] = xywh2xyxy(target_bboxes[..., 1:])
            pred_bboxes[..., 1:] = xywh2xyxy(pred_bboxes[..., 1:])

            non_suppressed_bboxes_idxs = nms(boxes=pred_bboxes[..., 1:], scores=pred_bboxes[..., 0], iou_threshold=self.thr)
            non_suppressed_bboxes = pred_bboxes[non_suppressed_bboxes_idxs]

            iou_scores = box_iou(target_bboxes[..., 1:], non_suppressed_bboxes[..., 1:])
            max_iou_scores = torch.max(iou_scores, dim=1)

            self.total_score += torch.sum(max_iou_scores.values)
            self.total_samples += torch.sum(torch.any(iou_scores, dim=1))

    def compute(self):
        numerator = self.total_score
        denominator = self.total_samples
        return 0. if denominator == 0 else numerator / denominator

    def reset(self):
        self.total_samples = 0
        self.total_score = 0


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
        return 0. if denominator == 0 else numerator / denominator

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
        return 0. if denominator == 0 else numerator / denominator

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
        return (
            dict.fromkeys(self._loss_sum_dict.keys(), 0)
            if self._total == 0
            else {
                loss_name: loss_value / self._total
                for loss_name, loss_value in self._loss_sum_dict.items()
            }
        )

    def reset(self):
        self._loss_sum_dict = defaultdict(lambda: 0)
        self._total = 0


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy", default_value=0)
        self.total_correct = 0
        self.total = 0

    def step(self, y, y_pred):
        y, y_pred = self.prepare(y, y_pred)

        correct = torch.eq(y, y_pred)

        self.total_correct += torch.sum(correct).item()
        self.total += correct.shape[0]

    def compute(self):
        return self.total_correct / self.total

    def reset(self):
        self.total_correct = 0
        self.total = 0
