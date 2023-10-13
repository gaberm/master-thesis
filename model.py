from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(cfg: DictConfig):
    model_name = cfg.model.name
    match model_name:
        case "mBERT":
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint=cfg.model.checkpoint,
                num_labels=cfg.model.num_labels)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint=cfg.model.checkpoint)
            return model, tokenizer