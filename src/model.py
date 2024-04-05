from transformers import AutoTokenizer, AutoModelForSequenceClassification
import adapters
import torch.nn as nn
import os
import numpy as np

def load_model(config):
    source_lang = config.params.source_lang
    has_lang_adapter = False
    has_task_adapter = False
    if "madx" in config.keys():
        has_lang_adapter = "lang_adapter" in config.madx.keys()
        has_task_adapter = "task_adapter" in config.madx.keys()
    load_ckpt = config.model.load_ckpt
    ckpt_averaging = config.model.ckpt_averaging

    if "copa" in config.trainer.exp_name:
        if config.model.name == "xlmr":
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model.hf_path,
            )
            model.classifier = CopaClassifier(
                input_dim=model.config.hidden_size,
                hidden_dim=model.config.hidden_size,
                output_dim=2,
                dropout_prob=config.model.dropout
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model.hf_path,
                num_labels=1
            )
            model.classifier = CopaClassifier(
                input_dim=model.config.hidden_size,
                hidden_dim=model.config.hidden_size,
                output_dim=1,
                dropout_prob=config.model.dropout
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.hf_path,
            num_labels=config.model.num_labels
        )
        if ckpt_averaging:
            average_state_dict = get_ckpt_average(config)
            model.load_state_dict(average_state_dict)
        

    if has_lang_adapter or has_task_adapter:
        # we must call adapters.init() to load adapters
        adapters.init(model)

        if has_lang_adapter:
            lang_adapter_cfg = adapters.AdapterConfig.load(
                "pfeiffer",
                non_linearity="gelu",
                reduction_factor=2
            )
            # we use pre-trained language adapters
            for path in config.madx.lang_adapter.values():
                _ = model.load_adapter(path, lang_adapter_cfg)

        # we add an untrained task adapter
        # for train, we train the task adapter for the task
        # for test, we load the weights of the task adapter from the best checkpoint
        task_adapter_name = config.madx.task_adapter.name
        model.add_adapter(task_adapter_name, config="seq_bn")
       
        if not load_ckpt:
            # train_adapter freezes the weights of the model 
            # and the language adapters to prevent them from further finetuning
            model.train_adapter([task_adapter_name]) 

            # active_adapters are the adapters that are used in the forward pass
            # if we use language adapter, we stack the task adapter on top of the language adapter
            # https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/04_Cross_Lingual_Transfer.ipynb
            if has_lang_adapter:
                model.active_adapters = adapters.Stack(source_lang, task_adapter_name)
            else:
                model.active_adapters = task_adapter_name

    return model    


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.hf_path)
 

# we use the same classifier for xcopa that the authors used in their paper
# https://arxiv.org/pdf/2005.00333.pdf
class CopaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(CopaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    

def get_ckpt_average(config):
    ckpt_dir = f"{config.data_dir}{config.model.ckpt_path}"
    ckpt_files = []
    for filename in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, filename)):
            ckpt_files.append(filename)
    
    k = len(ckpt_files)
    average_state_dict = {np.load(ckpt_files[0], allow_pickle=True)}
    for key, value in average_state_dict.items():
        if value.is_floating_point():
            average_state_dict[key] = value / k

    for ckpt_file in ckpt_files[1:]:
        ckpt = np.load(ckpt_file, allow_pickle=True)
        for key, value in ckpt.items():
            if value.is_floating_point():
                average_state_dict[key] += value / k

    return average_state_dict