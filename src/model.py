from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithHeads

def load_model(config: DictConfig) -> tuple[AutoModelForSequenceClassification | AutoModelWithHeads, AutoTokenizer]:
    name_or_path = config.model.load_args.name_or_path
    lang_adapter = config.adapter.lang_adapter
    task_adapter = config.adapter.task_adapter
    # load model with adapters
    if (lang_adapter != None) | (task_adapter != None):
        model = AutoModelWithHeads.from_pretrained(name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        # load language adapters
        if lang_adapter != None:
            for lang, adapter in config.adapter.lang_adapter.lang_adapter_heads.items():
                adapter_name = model.load_adapter(adapter)
                # for training we only want to use the adapter for the source language
                if lang == config.params.source_lang:
                    model.set_active_adapters(adapter_name)
        # load task adapters
        if task_adapter != None:
            name_or_path = config.adapter.task_adapter.load_args.name_or_path
            adapter_name = model.load_adapter(name_or_path)
            model.set_active_adapters(adapter_name)
            model.train_adapter(adapter_setup=[adapter_name])
    # load model without adapters
    else:
        model = AutoModelForSequenceClassification.from_pretrained(**config.model.load_args)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return model, tokenizer