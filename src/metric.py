from omegaconf import DictConfig
import torch
from torchmetrics.classification import F1Score, MulticlassCalibrationError, BinaryCalibrationError
                            

def load_metric(config: DictConfig) -> (F1Score | MulticlassCalibrationError | BinaryCalibrationError):
    metric_name = config.params.metric
    num_classes = config.model.load_args.num_classes

    if metric_name == "f1":
        metric = F1Score(num_classes=num_classes, average="micro")

    if metric_name == "ece":
        if num_classes == 2:
            metric = BinaryCalibrationError(n_bins=10, norm="l1")
        else:
            metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm="l1")

    else:
        raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric