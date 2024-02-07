from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
import platform
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(config.model.hf_path, num_labels=config.model.num_labels)

    # if madx is specified, load it
    if "madx" in config.keys():
        adapters.init(model)
        # load pretrained language adapters
        for hf_path in config.madx.lang_adapter[config.model.name].values():
            _ = model.load_adapter(hf_path)
        # create task adapter for training
        task_adapter_name = config.madx.task_adapter.name
        madx_config = adapters.SeqBnConfig(reduction_factor=config.madx.task_adapter.reduction_factor)    
        model.add_adapter(task_adapter_name, madx_config)
        model.active_adapters = adapters.Stack(source_lang, task_adapter_name)
        # train_adapter freezes the weights of the model and the language adapters to prevent them from further finetuning
        # however the language adapter is active and is used in the forward pass
        model.train_adapter([task_adapter_name])

    # if lora is specified, load it
    if "lora" in config.keys():
        cfg = LoraConfig(**config.lora)
        model = get_peft_model(model, cfg)

    return model


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)


def set_task_adapter_name(config): 
    try: 
        return config.madx.task_adapter.name 
    except: 
        return None