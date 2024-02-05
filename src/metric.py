from omegaconf import DictConfig
from torchmetrics.classification import F1Score, Accuracy, MulticlassCalibrationError, BinaryCalibrationError
import platform
                            

def load_metric(config, type):
    num_labels = config.model.num_labels

    if type == "val":
        metric_name = config.params.val_metric
    elif type == "uncertainty":
        metric_name = config.params.uncertainty_metric
    else:
        raise ValueError(f"Metric type {type} not supported.")
    
    # device is set to "mps:0" for Mac OS and "cuda" for Linux
    if platform.system() == "Darwin":
        device = "mps:0"
    else:
        device = "cuda"

    if type == "val":
        if metric_name == "f1":
            if num_labels == 2:
                metric = F1Score(task="binary").to(device)
            else:
                metric = F1Score(task="multiclass", num_classes=num_labels, average="micro").to(device)
        elif metric_name == "accuracy":
            if num_labels == 2:
                metric = Accuracy(task="binary").to(device)
            else:
                metric = Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device)
        else:
            raise ValueError(f"Metric {metric_name} not supported.")
    else:
        if metric_name == "expected_calibration_error":
            if num_labels == 2:
                metric = BinaryCalibrationError(n_bins=10, norm="l1").to(device)
            else:
                metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to(device)
        elif metric_name == "maximum_calibration_error":
            if num_labels == 2:
                metric = BinaryCalibrationError(n_bins=15, norm="max")
            else:
                metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="max")
        else:
            raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric
