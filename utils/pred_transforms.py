prediction_transforms_dict = {
    'accuracy': lambda x: (x > 0).float(),
    'recall': lambda x: (x > 0).float(),
    'precision': lambda x: (x > 0).float()
}
