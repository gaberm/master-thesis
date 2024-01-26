from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang
    model_name = config.model.name
    model_path = config.model.path

    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.model.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # if madx is specified, load it
    if "madx" in config.keys():
        adapters.init(model)
        # load pretrained language adapters
        for path in config.madx.lang_adapter[model_name].values():
            _ = model.load_adapter(path)
        # create task adapter for training
        task_adapter_name = config.madx.task_adapter.name    
        model.add_adapter(task_adapter_name, config="seq_bn")
        model.train_adapter([task_adapter_name])
        model.active_adapters = adapters.Stack(source_lang, task_adapter_name)

    # if lora is specified, load it
    if "lora" in config.keys():
        cfg = LoraConfig(**config.lora)
        model = get_peft_model(model, cfg)

    return model, tokenizer