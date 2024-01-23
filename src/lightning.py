from lightning import LightningModule
import torch
from omegaconf import DictConfig
from .optimizer import load_optimizer
from .metric import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import adapters

class LModel(LightningModule):
    def __init__(self, model: tuple[AutoModelForSequenceClassification, AutoTokenizer], config: DictConfig, adapter_names: dict[str, str]):
        super().__init__()
        self.model = model
        self.metric = load_metric(config)
        self.metric_name = config.params.metric
        self.source_lang = config.params.source_lang
        self.target_lang = ""
        self.optimizer = config.params.optimizer
        self.lr = config.params.lr
        self.adapter_names_dict = adapter_names
        self.label_smoothing = config.params.label_smoothing

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        batch_data = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch_data)
        loss = outputs.loss
        if self.label_smoothing > 0:
            loss = torch.nn.CrossEntropyLoss(outputs.logits, batch_data["labels"], label_smoothing=self.label_smoothing)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        probas = torch.softmax(logits, dim=-1)
        self.metric.update(probas, batch["labels"])
    
    def on_validation_epoch_end(self):
        val_score = self.metric.compute()
        self.log(f"Validation {self.metric_name} Score: ", val_score, prog_bar=True)

    def on_test_epoch_start(self):
        # activate adapters for zero-shot cross lingual transfer
        if "task_adapter" in self.adapter_names_dict and self.target_lang in self.adapter_names_dict[self.target_lang]:
            self.model.activate_adapters(None)
            self.model.active_adapters(adapters.Stack(self.adapter_names_dict[self.target_lang], self.adapter_names_dict["task_adapter"]))
        elif self.target_lang in self.adapter_names_dict:
            self.model.activate_adapters(None)
            self.model.active_adapters(self.adapter_names_dict[self.target_lang])
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        probas = torch.softmax(logits, dim=-1)
        self.metric.update(probas, batch["labels"])

    def on_test_epoch_end(self):
        test_score = self.metric.compute()
        self.log(f"Test {self.metric_name} Score for {self.target_lang}: ", test_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        return optimizer