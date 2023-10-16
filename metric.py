import evaluate
from omegaconf import DictConfig

def load_metric(cfg: DictConfig):
    metric_name = cfg.params.metric
    match metric_name:
        case "f1":
            f1 = evaluate.load("f1")
            return f1
