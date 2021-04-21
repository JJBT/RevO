prediction_transforms_dict = {
    'accuracy_prediction': lambda x: (x[..., 0] > 0).float(),
    'accuracy_target': lambda x: (x[..., 0] > 0).float(),
    'recall_prediction': lambda x: (x[..., 0] > 0).float(),
    'recall_target': lambda x: (x[..., 0] > 0).float(),
    'precision_prediction': lambda x: (x[..., 0] > 0).float(),
    'precision_target': lambda x: (x[..., 0] > 0).float()
}
