from torchmetrics.classification import F1Score, Accuracy, MulticlassCalibrationError, BinaryCalibrationError
import platform
                            

def load_metric(config, metric_type):
    num_labels = config.model.num_labels

    match metric_type:
        case "pred":
            metric_name = config.params.pred_metric
        case "uncert":
            metric_name = config.params.uncert_metric
        case _:
            raise ValueError(f"Metric type {metric_type} not supported.")
    
    # device is set to "mps:0" for Mac and "cuda" for Linux
    device = config.trainer.gpu_name[platform.system().lower()]

    match (metric_type, metric_name, num_labels):
        case ("pred", "f1", 2):
            metric = F1Score(task="binary").to(device)
        case ("pred", "f1", _):
            metric = F1Score(task="multiclass", num_classes=num_labels, average="micro").to(device)
        case ("pred", "accuracy", 2):
            metric = Accuracy(task="binary").to(device)
        case ("pred", "accuracy", _):
            metric = Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device)
        case ("uncert", "ece", 2):
            metric = BinaryCalibrationError(n_bins=10, norm="l1").to(device)
        case ("uncert", "ece", _):
            metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to(device)
        case ("uncert" "mce", 2):
            metric = BinaryCalibrationError(n_bins=10, norm="max").to(device)
        case ("uncert", "mce", _):
            metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="max").to(device)
        case (_, _, _):
            raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric
