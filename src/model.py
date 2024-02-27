from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import adapters
import torch
import os
import re
from peft import get_peft_model, LoraConfig
from src.utils import get_best_checkpoint


def load_model(config):
    source_lang = config.params.source_lang
    using_madx = "madx" in config.keys()
    test_run = "ckpt_dir" in config.model.keys()
    using_lora = "lora" in config.keys()

    if using_madx:
        task_adapter_name = config.madx.task_adapter.name

        model = adapters.AutoAdapterModel.from_pretrained(config.model.hf_path)
        model.add_multiple_choice_head(task_adapter_name, num_choices=2)

        # we must call adapters.init() to load adapters
        adapters.init(model)
        
        # we use pre-trained language adapters for cross-lingual transfer
        for path in config.madx.lang_adapter.values():
            lang_adapter_cfg = adapters.AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=2)
            _ = model.load_adapter(path, lang_adapter_cfg)

        if test_run:
            ckpt_dir = get_best_checkpoint(config.model.ckpt_dir)
            model.load_adapter(ckpt_dir, task_adapter_name)
            model.load_head(ckpt_dir, task_adapter_name)
        else:
            model.add_adapter(task_adapter_name, config="seq_bn")
       
            # train_adapter freezes the weights of the model 
            # and the language adapters to prevent them from further finetuning
            model.train_adapter([task_adapter_name]) 

            # active_adapters are the adapters that are used in the forward pass
            # for mad-x, we stack the task adapter on top of the language adapter
            # https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/04_Cross_Lingual_Transfer.ipynb
            model.active_adapters = adapters.Stack(source_lang, task_adapter_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config.model.hf_path, num_labels=config.model.num_labels)
    
        # load model checkpoint for testing
        if test_run:
            ckpt_path = get_best_checkpoint(config.model.ckpt_dir)
            state_dict = torch.load(ckpt_path, map_location="cuda:0")["state_dict"]
            # replace model. with an empty string to match the keys of the model
            model_ckpt = {k.replace("model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(model_ckpt)
        
        # if using_lora:
        #     lora_cfg = LoraConfig(**config.lora)
        #     model = get_peft_model(model, lora_cfg)

    return model    


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)
 