from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
import torch
import platform
from peft import get_peft_model, LoraConfig

def load_model(config):
    source_lang = config.params.source_lang

    model = AutoModelForSequenceClassification.from_pretrained(config.model.hf_path, num_labels=config.model.num_labels)
    # load model checkpoint for testing
    if config.model.load_ckpt:
        ckpt_path = f"{config.data_dir[platform.system()]}checkpoints/{config.trainer.exp_name}/{config.model.ckpt}"
        # replace model. with an empty string to match the keys of the model
        ckpt = {k.replace("model.", ""): v for k, v in torch.load(ckpt_path)["state_dict"].items()}
        model.load_state_dict(ckpt)

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
        # https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
        model.active_adapters = adapters.Stack(source_lang, task_adapter_name)

    # if lora is specified, load it
    if "lora" in config.keys():
        cfg = LoraConfig(**config.lora)
        model = get_peft_model(model, cfg)

    return model


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)