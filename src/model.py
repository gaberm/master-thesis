from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
from peft import get_peft_model, LoraConfig

def load_model(config: DictConfig) -> (AutoModelForSequenceClassification, AutoTokenizer):
    source_lang = config.params.source_lang
    model_name = config.model.name
    model_path = config.model.path
    # check if adapters or lora are specified
    try:
        lang_adapter_paths = config.lang_adapter
    except:
        lang_adapter_paths = None
    try:
        task_adapter_paths = config.task_adapter
    except:
        task_adapter_paths = None
    try:
        lora_config = config.lora
    except:
        lora_config = None

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

    # if lora is specified, load it
    if lora_config is not None:
        cfg = LoraConfig(**lora_config)
        model = get_peft_model(model, cfg)

    return model, tokenizer, adapter_name_dict