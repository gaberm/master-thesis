from lightning import LightningModule
import adapters
import torch
import platform
from transformers import get_linear_schedule_with_warmup
from .optimizer import load_optimizer
from .metric import load_metric

class LModel(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.pred_metric = load_metric(config, "pred")
        self.pred_metric_name = config.params.pred_metric
        self.uncert_metric = load_metric(config, "uncert")
        self.uncert_metric_name = config.params.uncert_metric
        self.source_lang = config.params.source_lang
        self.target_lang = ""
        self.task_adapter_name = config.madx.task_adapter.name if "madx" in config.keys() else None
        self.optimizer = config.params.optimizer
        self.lr = config.params.lr
        self.num_labels = config.model.num_labels
        self.label_smoothing = config.params.label_smoothing
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.data_dir = config.data_dir[platform.system().lower()]
        self.save_hyperparameters(ignore=["model"])

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
        preds = outputs.logits.argmax(dim=-1)
        self.pred_metric.update(preds, batch["labels"])
    
    def on_validation_epoch_end(self):
        val_score = self.pred_metric.compute()
        self.log(f"{self.pred_metric_name}", val_score, prog_bar=True)
        self.pred_metric.reset()
        # save the adapter for each checkpoint
        if self.task_adapter_name is not None:
            try:
                adapter_dir = f"{self.data_dir}checkpoints/latest-run/epoch={self.trainer.current_epoch}-step={self.trainer.global_step}-{self.pred_metric_name}={val_score:.3f}"
                self.model.save_adapter(adapter_dir, self.task_adapter_name, with_head=True)
            except FileExistsError:
                pass

    def on_test_epoch_start(self):
        # activate target_lang adapter for zero-shot cross-lingual transfer
        if self.task_adapter_name is not None:
            self.model.set_active_adapters(None)
            self.model.active_adapters = adapters.Stack(self.target_lang, self.task_adapter_name)
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        preds = outputs.logits.argmax(dim=-1)
        probas = torch.softmax(logits, dim=-1)
        if self.num_labels == 2:
            probas = probas[:, 1]
        self.uncert_metric.update(probas, batch["labels"])
        self.pred_metric.update(preds, batch["labels"])

    def on_test_epoch_end(self):
        uncert_score = self.uncert_metric.compute()
        pred_score = self.pred_metric.compute()
        self.log(f"{self.uncert_metric_name} {self.target_lang}", uncert_score, prog_bar=True)
        self.log(f"{self.pred_metric_name} {self.target_lang}", pred_score, prog_bar=True)
        # reset metrics for the next target language
        self.uncert_metric.reset()
        self.pred_metric.reset()

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        # we don't use a scheduler for mad-x,
        # because the authors of the mad-x paper don't use one
        if self.task_adapter_name is None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.trainer.estimated_stepping_batches*0.01, 
                num_training_steps=self.trainer.estimated_stepping_batches)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

        
