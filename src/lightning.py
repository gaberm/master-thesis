from lightning import LightningModule
import torch
from omegaconf import DictConfig
from .optimizer import load_optimizer
from .metric import load_metric
from .model import set_task_adapter_name
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import adapters

class LModel(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.val_metric = load_metric(config, "val")
        self.uncertainty_metric = load_metric(config, "uncertainty")
        self.source_lang = config.params.source_lang
        self.target_lang = ""
        self.task_adapter_name = set_task_adapter_name(config)
        self.optimizer = config.params.optimizer
        self.lr = config.params.lr
        self.num_labels = config.model.num_labels
        self.label_smoothing = config.params.label_smoothing
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        batch_data = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch_data)
        loss = outputs.loss
        if self.label_smoothing > 0:
            loss = self.ce_loss(outputs.logits, batch_data["labels"])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits.argmax(dim=-1)
        self.val_metric.update(logits, batch["labels"])
    
    def on_validation_epoch_end(self):
        val_score = self.val_metric.compute()
        self.log(f"{self.val_metric}: ", val_score, prog_bar=True)

    def on_test_epoch_start(self):
        # activate target_lang adapter for zero-shot cross lingual transfer
        if self.task_adapter_name is not None:
            self.model.active_adapters = adapters.Stack(self.target_lang, self.task_adapter_name)
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        probas = torch.softmax(logits, dim=-1)
        if self.num_labels == 2:
            probas = probas[:, 1]
        self.uncertainty_metric.update(probas, batch["labels"])

    def on_test_epoch_end(self):
        test_score = self.uncertainty_metric.compute()
        self.log(f"{self.uncertainty_metric} {self.target_lang}: ", test_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        return optimizer