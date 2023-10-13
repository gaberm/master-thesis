import torch.optim as optim
from omegaconf import DictConfig

def create_optimizer(model, cfg: DictConfig):
    if cfg.train.optimizer == "AdamW":
        return optim.AdamW(params=model.parameters(), lr=cfg.train.learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
