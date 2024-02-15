from omegaconf import DictConfig
from torchmetrics.classification import F1Score, Accuracy, MulticlassCalibrationError, BinaryCalibrationError
import platform
                            

def load_metric(config, metric_type):
    num_labels = config.model.num_labels

    match metric_type:
        case "val":
            metric_name = config.params.val_metric
        case "uncertainty":
            metric_name = config.params.uncertainty_metric
        case _:
            raise ValueError(f"Metric type {metric_type} not supported.")
    
    # device is set to "mps:0" for Mac OS and "cuda" for Linux
    device = config.trainer.gpu_name[platform.system()]

    match (metric_type, metric_name, num_labels):
        case ("val", "f1", 2):
            metric = F1Score(task="binary").to(device)
        case ("val", "f1", _):
            metric = F1Score(task="multiclass", num_classes=num_labels, average="micro").to(device)
        case ("val", "accuracy", 2):
            metric = Accuracy(task="binary").to(device)
        case ("val", "accuracy", _):
            metric = Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device)
        case ("uncertainty", "expected_calibration_error", 2):
            metric = BinaryCalibrationError(n_bins=10, norm="l1").to(device)
        case ("uncertainty", "expected_calibration_error", _):
            metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to(device)
        case ("uncertainty" "maximum_calibration_error", 2):
            metric = BinaryCalibrationError(n_bins=15, norm="max").to(device)
        case ("uncertainty", "maximum_calibration_error", _):
            metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="max").to(device)
        case (_, _, _):
            raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric
