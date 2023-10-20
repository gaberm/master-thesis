from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterConfig

def load_model(cfg: DictConfig):
    checkpoint = cfg.model.load_args.checkpoint
    print()
    if cfg.model.adapter_active:
        auto_config = AutoConfig.from_pretrained(checkpoint)
        model = AutoAdapterModel.from_pretrained(checkpoint, config=auto_config)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(cfg)
        lang_adapter_config = AdapterConfig.load(**cfg.model.lang_adapter_args)
        for checkpoint in cfg.model.lang_checkpoints:
            model.load_adapter(checkpoint, config=lang_adapter_config)
        model.add_adapter(cfg.model.task_adapter_args.head_name)
        model.add_multiple_choice_head(**cfg.model.task_adapter_args)
        model.train_adapter(adapter_setup=[cfg.model.task_adapter_args.head_name])
        model.set_active_adapters(cfg.model.task_adapter_args.head_name,
                                  cfg.params.source_lang)

        return model, tokenizer
    else:
        model = AutoModelForSequenceClassification.from_pretrained(**cfg.model.load_args)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return model, tokenizer