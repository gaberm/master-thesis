from torchmetrics.classification import Accuracy, MulticlassCalibrationError, BinaryCalibrationError
from src.utils import get_device

def load_metric(config, metric_type, num_labels=None):
    if metric_type == "pred":
        metric_name = config.params.pred_metric
    elif metric_type == "uncert":
        metric_name = config.params.uncert_metric
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}. Must be 'pred' or 'uncert'.")
    
    if num_labels is None:
        num_labels = config.model.num_labels
    device = get_device(config)
    
    if metric_type == "pred":
        if metric_name == "accuracy":
            if num_labels == 2:
                metric = Accuracy(task="binary").to(device)
            else:
                metric = Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device)
        else:
            raise ValueError(f"Unsupported metric name for prediction: {metric_name}")
    else:
        if metric_name == "ece":
            if num_labels == 2:
                metric = BinaryCalibrationError(n_bins=10, norm="l1").to(device)
            else:
                metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to(device)
        elif metric_name == "mce":
            if num_labels == 2:
                metric = BinaryCalibrationError(n_bins=10, norm="max").to(device)
            else:
                metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="max").to(device)
        else:
            raise ValueError(f"Unsupported metric name for uncertainty: {metric_name}")

    return metric
