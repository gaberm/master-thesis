from torchmetrics.classification import Accuracy, MulticlassCalibrationError, BinaryCalibrationError

def load_metric(config, metric_type, num_labels=None):
    if num_labels is None:
        num_labels = config.model.num_labels
        
    if metric_type == "pred":
        metric_name = config.params.pred_metric
    elif metric_type == "uncert":
        metric_name = config.params.uncert_metric
    else:
        raise ValueError(f"Invalid metric type: {metric_type}. Must be 'pred' or 'uncert'.")
    
    # device is set to "mps:0" for Mac and "cuda" for Linux
    device = config.trainer.gpu_name

    metrics = {"pred":{
        "accuracy": Accuracy(task="binary") if num_labels == 2 else Accuracy(task="multiclass", num_classes=num_labels, average="micro").to(device),
        },
        "uncert":{
            "ece": BinaryCalibrationError(n_bins=10, norm="l1").to(device) if num_labels == 2 else MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="l1").to(device),
            "mce": BinaryCalibrationError(n_bins=10, norm="max").to(device) if num_labels == 2 else MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm="max").to(device)    
        }
    }
    metric = metrics[metric_type][metric_name]
    
    return metric
