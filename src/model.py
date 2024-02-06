from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang
    model_path = config.model.path

    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.model.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # if madx is specified, load it
    if "madx" in config.keys():
        adapters.init(model)
        # load pretrained language adapters
        for path in config.madx.lang_adapter[config.model.name].values():
            _ = model.load_adapter(path)
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

    return model, tokenizer


def set_task_adapter_name(config): 
        try: 
            return config.madx.task_adapter.name 
        except: 
            return None