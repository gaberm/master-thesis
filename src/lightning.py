import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from .optimizer import create_optimizer
from .metric import load_metric
from transformers import AutoModelForSequenceClassification, AutoModelWithHeads, AutoTokenizer
from transformers.adapters.composition import Stack

class LightningModel(pl.LightningModule):
    def __init__(self, model: tuple[AutoModelForSequenceClassification | AutoModelWithHeads, AutoTokenizer], config: DictConfig):
        super().__init__()
        self.model = model
        self.metric = load_metric(config)
        self.metric_name = config.params.metric
        self.lang_adapters = self.set_lang_adapter(config)
        self.task_adapter = self.set_task_adapter(config)
        self.source_lang = config.params.source_lang
        self.target_lang = ""
        self.conf_lst = []
        self.label_lst = []
    
    # setter method for language adapters
    def set_lang_adapter(self, config: DictConfig): 
        if config.adapters.lang_adapter != None:
            lang_adapter = config.adapters.lang_adapter.lang_adapter_heads
            return lang_adapter
        else:
            return None
    
    # setter method for task adapter
    def set_task_adapter(self, config: DictConfig):
        if config.adapter.task_adapter != None:
            task_adapter = config.adapters.task_adapter.load_args.name_or_path
            return task_adapter
        else:
            return None
    
    # activate task and target language adapters
    def activate_adapters(self, target_lang: str):
        self.target_lang = target_lang
        self.model.set_active_adapters(None)
        if self.task_adapter != None and self.lang_adapters != None:
            self.model.set_active_adapters(Stack(self.task_adapter, self.lang_adapters[target_lang]))
        elif self.lang_adapters != None:
            self.model.set_active_adapters(self.lang_adapters[target_lang])
        elif self.task_adapter != None:
            self.model.set_active_adapters(self.task_adapter)
            print("Warning: No language adapters were loaded. Only the task adapter is active.")

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        batch_data = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch_data)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        confidences = torch.softmax(logits, dim=-1)
        self.conf_lst.append(confidences)
        self.label_lst.append(batch["labels"])
    
    def on_validation_epoch_end(self):
        conf_tensor = torch.stack(self.prediction_lst)
        label_tensor = torch.cat(self.label_lst)
        val_score = self.metric(conf_tensor, label_tensor)
        self.log(f"{self.metric_name}", val_score)
        self.conf_lst = []
        self.label_lst = []
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        confidences = torch.softmax(logits, dim=-1)
        self.conf_lst.append(confidences)
        self.label_lst.append(batch["labels"])

    def on_test_epoch_end(self):
        conf_tensor = torch.stack(self.conf_lst)
        label_tensor = torch.cat(self.label_lst)
        test_score = self.metric(conf_tensor, label_tensor)
        self.log(f'{self.metric_name}_{self.target_lang}', test_score)
        self.conf_lst = []
        self.label_lst = []

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        return optimizer