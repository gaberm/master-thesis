from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithHeads

def load_model(cfg: DictConfig) -> tuple[AutoModelForSequenceClassification | AutoModelWithHeads, AutoTokenizer]:
    model_name = cfg.model.load_args.pretrained_model_name_or_path
    if cfg.model.adapter_active:
        model = AutoModelWithHeads.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for lang in cfg.model.lang_adapter_heads.keys():
            adapter_name = model.load_adapter(cfg.model.lang_adapter_heads[lang])
            if lang == cfg.params.source_lang:
                model.set_active_adapters(adapter_name)
        adapter_name = model.load_adapter(**cfg.model.task_adapter_args)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_setup=[adapter_name])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(**cfg.model.load_args)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer