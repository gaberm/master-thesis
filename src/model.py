from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
import torch
import platform
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang
    using_madx = "madx" in config.keys()
    using_ckpt = "load_path" in config.model.keys()
    using_lora = "lora" in config.keys()

    model = AutoModelForSequenceClassification.from_pretrained(config.model.hf_path, num_labels=config.model.num_labels)
    
    # load model checkpoint for testing
    if using_ckpt and not using_madx:
        # replace model. with an empty string to match the keys of the model
        model_ckpt = {k.replace("model.", ""): v for k, v in torch.load(config.model.ckpt_path)["state_dict"].items()}
        model.load_state_dict(model_ckpt)

    if using_madx:
        # we must call adapters.init() to load adapters
        adapters.init(model)
        
        # we use pre-trained language adapters for cross-lingual transfer
        for path in config.madx.lang_adapter[config.model.name].values():
            _ = model.load_adapter(path)
        
        task_adapter_name = config.madx.task_adapter.name
        if using_ckpt:
            model_ckpt = torch.load(config.model.ckpt_path)
            task_adapter_config = adapters.SeqBnConfig.from_dict(model_ckpt["state_dict"])
            model.add_adapter(task_adapter_name, task_adapter_config)
        else:
            madx_config = adapters.SeqBnConfig(reduction_factor=config.madx.task_adapter.reduction_factor)    
            model.add_adapter(task_adapter_name, madx_config)
       
            # train_adapter freezes the weights of the model 
            # and the language adapters to prevent them from further finetuning
            model.train_adapter([task_adapter_name]) 

            # active_adapters are the adapters that are used in the forward pass
            # for mad-x, we stack the task adapter on top of the language adapter
            # https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
            model.active_adapters = adapters.Stack(source_lang, task_adapter_name)
        
    if using_lora:
        cfg = LoraConfig(**config.lora)
        model = get_peft_model(model, cfg)

    return model


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)