import evaluate
from omegaconf import DictConfig

def load_metric(cfg: DictConfig) -> evaluate.EvaluationModule:
    metric_name = cfg.params.metric
    if metric_name == "f1":
        metric = evaluate.load("f1")
    return metric
