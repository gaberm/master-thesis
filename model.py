from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(cfg: DictConfig):
    model_name = cfg.model.name
    match model_name:
        case "mBERT":
            print(cfg.model.checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.model.checkpoint, num_labels=cfg.model.num_labels)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
            return model, tokenizer