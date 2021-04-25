from utils.data import from_yolo_target_torch
import torch


transforms_dict = {
    'accuracy_prediction': lambda x: (x[..., 0] > 0).float(),
    'accuracy_target': lambda x: (x[..., 0] > 0).float(),
    'recall_prediction': lambda x: (x[..., 0] > 0).float(),
    'recall_target': lambda x: (x[..., 0] > 0).float(),
    'precision_prediction': lambda x: (x[..., 0] > 0).float(),
    'precision_target': lambda x: (x[..., 0] > 0).float(),
    'ap_prediction': lambda x: from_yolo_target_torch(torch.sigmoid(x), [320, 320], [10, 10]),
    'ap_target': lambda x: from_yolo_target_torch(x, [320, 320], [10, 10]),
    'iou_prediction': lambda x: from_yolo_target_torch(torch.sigmoid(x), [320, 320], [10, 10]),
    'iou_target': lambda x: from_yolo_target_torch(x, [320, 320], [10, 10]),
}



