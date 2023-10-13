from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(cfg: DictConfig):
    if cfg.model == "mBERT":
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint=cfg.model.checkpoint,
            num_labels=cfg.model.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint=cfg.model.checkpoint)
        return model, tokenizer
    else:
        raise ValueError("Unsupported optimizer")