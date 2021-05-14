from datasets.dataset import ObjectDetectionDataset, ObjectClassificationDataset
from datasets.dataset import object_classification_collate_fn as classification_collate_fn
from datasets.dataset import object_detection_collate_fn as detection_collate_fn


def object_classification_collate_fn():
    return classification_collate_fn


def object_detection_collate_fn():
    return detection_collate_fn
