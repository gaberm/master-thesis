from omegaconf import DictConfig
from torchmetrics.classification import F1Score, Accuracy, MulticlassCalibrationError, BinaryCalibrationError
import platform
                            

def load_metric(metric_name, num_labels):
    # set device based on OS
    # mps:0 is the GPU device on MacOS
    # cuda is the GPU device on Linux
    op_system = platform.system()
    if op_system == "Darwin":
        device = "mps:0"
    else:
        device = "cuda"

    if metric_name == "f1":
        if num_labels == 2:
            metric = F1Score(task="binary").to(device)
        else:
            metric = F1Score(task="multiclass", num_classes=num_labels, average="micro").to(device)
    if metric_name == "accuracy":
        if num_labels == 2:
            metric = Accuracy(task="binary").to(device)
        else:
            metric = Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device)
    elif metric_name == "expected_calibration_error":
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