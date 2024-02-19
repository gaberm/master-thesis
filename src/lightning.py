from lightning import LightningModule
import adapters
import torch
import os
import platform
from .optimizer import load_optimizer
from .metric import load_metric

class LModel(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.val_metric = load_metric(config, "val")
        self.val_metric_name = config.params.val_metric
        self.uncertainty_metric = load_metric(config, "uncertainty")
        self.uncertainty_metric_name = config.params.uncertainty_metric
        self.source_lang = config.params.source_lang
        self.target_lang = ""
        self.task_adapter_name = config.madx.task_adapter.name if "madx" in config.keys() else None
        self.optimizer = config.params.optimizer
        self.lr = config.params.lr
        self.num_labels = config.model.num_labels
        self.label_smoothing = config.params.label_smoothing
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.best_val_score = 0
        self.ckpt_path = config.data_dir[platform.system()] + f"checkpoints/{config.trainer.exp_name}"
        self.save_hyperparameters()

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
        self.log(f"{self.val_metric_name}", val_score, prog_bar=True) 
        # save the best checkpoint
        # we must save checkpoints manually, 
        # because Lightning's checkpointing doesn't work for self-trained task adapters
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        if val_score > self.best_val_score:
            # The weights of the base-model are freezed during (adapter) training
            # Therefore, we only save the adapter weights
            if self.task_adapter_name is not None:
                self.model.save_adapter(self.ckpt_path+f"{{epoch}}-{self.val_metric_name}:{{{self.val_score}:.3f}}",
                                        self.set_task_adapter_name)

    def on_test_epoch_start(self):
        # activate target_lang adapter for zero-shot cross-lingual transfer
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
        self.log(f"{self.uncertainty_metric_name} {self.target_lang}", test_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        return optimizer