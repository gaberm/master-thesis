from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters

def load_model(config: DictConfig) -> (AutoModelForSequenceClassification, AutoTokenizer):
    model_name = config.model.name
    model_path = config.model.path
    lang_adapter_paths = config.lang_adapter
    task_adapter_paths = config.task_adapter
    source_lang = config.params.source_lang

    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.model.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    adapter_name_dict = {}
    # if language adapters are specified, load them
    if lang_adapter_paths is not None or task_adapter_paths is not None:
        adapters.init(model)
        # load language adapters
        if lang_adapter_paths is not None:
            for lang, path in lang_adapter_paths[model_name].items():
                adapter_name = model.load_adapter(path)
                adapter_name_dict[lang] = adapter_name

        # load task adapter
        if task_adapter_paths is not None:
            adapter_path = config.task_adapter[model_name].path
            adapter_name = model.load_adapter(adapter_path)
            adapter_name_dict["task_adapter"] = adapter_name
        
        # activate adapters
        if lang_adapter_paths is not None and task_adapter_paths is not None:
            model.set_active_adapters(adapters.Stack(adapter_name_dict[source_lang], adapter_name_dict["task_adapter"]))
        elif lang_adapter_paths is not None:
            model.set_active_adapters(adapter_name_dict[source_lang])
        else:
            model.set_active_adapters(adapter_name_dict["task_adapter"])

    return model, tokenizer, adapter_name_dict