import evaluate
from omegaconf import DictConfig
import torch

class expectedCalibrationError:
    def __init__(self):
        self.prediction_data = []
        self.confidence_data = []
        self.label_data = []

    def add_batch(self, predictions, confidences, labels):
        self.prediction_data.append(predictions)
        self.confidence_data.append(confidences)
        self.label_data.append(labels)

    def compute(self, num_bins=10):
        bin_edges = torch.linspace(0, 1, num_bins + 1)

        ece = 0
        bin_confidences = torch.zeros(num_bins)
        bin_accuracies = torch.zeros(num_bins)

        predicted_labels = torch.cat(self.prediction_data)
        confidence_scores = torch.cat(self.confidence_data)
        labels = torch.cat(self.label_data)

        for i in range(num_bins):
            bin_mask = (confidence_scores >= bin_edges[i]) & (confidence_scores < bin_edges[i + 1])
            bin_correct = (predicted_labels == labels)[bin_mask]

            bin_accuracy = bin_correct.float().mean()
            bin_confidence = confidence_scores[bin_mask].mean()

            bin_accuracies[i] = bin_accuracy
            bin_confidences[i] = bin_confidence

            ece += torch.abs(bin_accuracy - bin_confidence) * bin_mask.float().mean()
    
        return ece, bin_accuracies, bin_confidences
                            

def load_metric(cfg: DictConfig) -> (evaluate.EvaluationModule | expectedCalibrationError):
    metric_name = cfg.params.metric

    if metric_name == "f1":
        metric = evaluate.load("f1")

    if metric_name == "ece":
        metric = expectedCalibrationError()

    else:
        raise ValueError(f"Metric {metric_name} not supported.")
    
    return metric