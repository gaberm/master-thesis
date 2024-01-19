from omegaconf import DictConfig
from torchmetrics.classification import F1Score, MulticlassCalibrationError, BinaryCalibrationError
                            

def load_metric(config: DictConfig) -> (F1Score | MulticlassCalibrationError | BinaryCalibrationError):
    metric_name = config.params.metric
    num_labels = config.model.num_labels

    if metric_name == "f1":
        metric = F1Score(task="multiclass", num_classes=num_labels, average="micro").to("mps:0")

    elif metric_name == "ece":
        if num_labels == 2:
            metric = BinaryCalibrationError(n_bins=10, norm="l1").to("mps:0")
        else:
            metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to("mps:0")

    else:
        raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric