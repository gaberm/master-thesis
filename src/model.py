from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang

    # load model from checkpoint for testing or from huggingface for training
    if config.model.load_ckpt:
        load_path = config.model.ckpt_path["state_dict"]
    else:
        load_path = config.model.hf_path
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=config.model.num_labels)

    if "madx" in config.keys():
        adapters.init(model)
        
        # we use pre-trained language adapters for cross-lingual transfer
        for path in config.madx.lang_adapter[config.model.name].values():
            _ = model.load_adapter(path)
        
        # we create an un-trained task adapter that we will train by ourselves
        task_adapter_name = config.madx.task_adapter.name
        madx_config = adapters.SeqBnConfig(reduction_factor=config.madx.task_adapter.reduction_factor)    
        model.add_adapter(task_adapter_name, madx_config)
       
        # train_adapter freezes the weights of the model 
        # and the language adapters to prevent them from further finetuning
        model.train_adapter([task_adapter_name])

        # active_adapters are the adapters that are used in the forward pass
        # for mad-x, we stack the task adapter on top of the language adapter
        model.active_adapters = adapters.Stack(source_lang, task_adapter_name)

    # if lora is specified, load it
    if "lora" in config.keys():
        cfg = LoraConfig(**config.lora)
        model = get_peft_model(model, cfg)

    return model


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)