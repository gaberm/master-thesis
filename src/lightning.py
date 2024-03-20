from lightning import LightningModule
import adapters
import torch
import platform
from transformers import get_scheduler
from .utils import compute_val_score
from .optimizer import load_optimizer
from .metric import load_metric
from .model import load_model

class LModel(LightningModule):
    def __init__(self, config, seed):
        super().__init__()
        self.model = load_model(config)
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
        self.exp_name = config.trainer.exp_name
        self.result_lst = []
        self.seed = seed
        self.save_hyperparameters()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = self.ce_loss(outputs.logits, batch["labels"])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        self.pred_metric.update(preds, batch["labels"])
    
    def on_validation_epoch_end(self):
        val_score = self.pred_metric.compute()
        self.log(f"{self.pred_metric_name}", val_score, prog_bar=True, sync_dist=True)
        self.pred_metric.reset()

    def on_test_epoch_start(self):
        # deactive all adapters and active the target language and task adapter
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
        self.result_lst.append(
            [self.exp_name,
             self.seed,
             self.target_lang, 
             self.uncert_metric_name,
             float(uncert_score)]
        )
        self.result_lst.append(
            [self.exp_name,
             self.seed,
             self.target_lang,
             self.pred_metric_name,
             float(pred_score)]
        )
        # reset metrics for the next target language
        self.uncert_metric.reset()
        self.pred_metric.reset()

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        # we don't use a scheduler for mad-x,
        # because the authors of the mad-x paper don't use one
        if self.task_adapter_name is None:
            scheduler = get_scheduler(
                "linear",
                optimizer, 
                num_warmup_steps=self.trainer.estimated_stepping_batches*0.1, 
                num_training_steps=self.trainer.estimated_stepping_batches)
            scheduler_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr_scheduler"
                }
            }
            return scheduler_dict
        else:
            return optimizer
        

class LModelCopa(LightningModule):
    def __init__(self, config, seed):
        super().__init__()
        self.model = load_model(config)
        self.pred_metric_binary = load_metric(config, "pred", 2)
        self.pred_metric_multiclass = load_metric(config, "pred", 3)
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
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.data_dir = config.data_dir[platform.system().lower()]
        self.exp_name = config.trainer.exp_name
        self.result_lst = []
        self.seed = seed
        self.copa_val_samples = 0
        self.siqa_val_samples = 0
        self.save_hyperparameters()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        # because we train the model on two datasets with a different number of labels 
        # (copa: 2, social_i_qa: 3) for xcopa, we have to handle the training and inference differently.
        # details can be found in the paper: https://arxiv.org/abs/2005.00333

        batches = {}
        for dataset, batch in batch.items():
            try:
                batches[dataset] = {k: v.to(self.device) for k, v in batch.items()}
            except AttributeError:
                pass

        outputs = {}
        for dataset, batch in batches.items():
            outputs[dataset] = self.model(**batch)

        loss = 0.0
        for dataset, output in outputs.items():
            idx = 0
            logit_lst = []
            label_lst = []
            if dataset == "copa":
                while idx < len(output.logits):
                    logit_lst.append(output.logits[idx:idx+2,1])
                    label_lst.append(batches[dataset]["labels"][idx:idx+2].argmax(dim=0))
                    if not batches[dataset]["labels"][idx:idx+2].eq(1).sum() == 1:
                        raise ValueError("More than one label is set to 1. This is not allowed.")
                    idx += 2
                num_rows = int(len(output.logits) / 2)
                logits = torch.cat(logit_lst).view(num_rows, 2)[:,1]
                labels = torch.stack(label_lst).to(torch.float32)

            if dataset == "siqa":
                while idx < len(output.logits):
                    logit_lst.append(output.logits[idx:idx+3,1])
                    label_lst.append(batches[dataset]["labels"][idx:idx+3].argmax(dim=0))
                    if not batches[dataset]["labels"][idx:idx+3].eq(1).sum() == 1:
                        raise ValueError("More than one label is set to 1. This is not allowed.")
                    idx += 3
                num_rows = int(len(output.logits) / 3)
                logits = torch.cat(logit_lst).view(num_rows, 3)
                labels = torch.stack(label_lst)

            loss = self.ce_loss(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batches = {}
        for dataset, batch in batch.items():
            try:
                batches[dataset] = {k: v.to(self.device) for k, v in batch.items()}
            except AttributeError:
                pass

        outputs = {}
        for dataset, batch in batches.items():
            outputs[dataset] = self.model(**batch)

        for dataset, output in outputs.items():
            idx = 0
            proba_lst = []
            label_lst = []

            if dataset == "copa":
                while idx < len(output.logits):
                    proba_lst.append(output.logits[idx:idx+2,1].softmax(dim=0))
                    label_lst.append(batches[dataset]["labels"][idx:idx+2].argmax(dim=0))
                    idx += 2
                num_rows = int(len(output.logits) / 2)
                probas = torch.cat(proba_lst).view(num_rows, 2)[:,1] 
                labels = torch.stack(label_lst)
                self.pred_metric_binary.update(probas, labels)
                self.copa_val_samples += len(labels)

            if dataset == "siqa":
                while idx < len(output.logits):
                    proba_lst.append(output.logits[idx:idx+3,1].softmax(dim=0))
                    label_lst.append(batches[dataset]["labels"][idx:idx+3].argmax(dim=0))
                    idx += 3
                num_rows = int(len(output.logits) / 3)
                probas = torch.cat(proba_lst).view(num_rows, 3)
                labels = torch.stack(label_lst)
                self.pred_metric_multiclass.update(probas, labels)
                self.siqa_val_samples += len(labels)

    
    def on_validation_epoch_end(self):
        val_score = compute_val_score(
            self.pred_metric_binary.compute(),
            self.pred_metric_multiclass.compute(),
            self.copa_val_samples,
            self.siqa_val_samples
        )
        self.log(f"{self.pred_metric_name}", val_score, prog_bar=True, sync_dist=True)
        self.pred_metric_binary.reset()
        self.pred_metric_multiclass.reset()
        self.copa_val_samples = 0
        self.siqa_val_samples = 0

    def on_test_epoch_start(self):
        # deactive all adapters and active the target language and task adapter
        if self.task_adapter_name is not None:
            self.model.set_active_adapters(None)
            self.model.active_adapters = adapters.Stack(self.target_lang, self.task_adapter_name)
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        idx = 0
        proba_lst = []
        label_lst = []
                
        while idx < len(outputs.logits):
            proba_lst.append(outputs.logits[idx:idx+2].softmax(dim=0))
            label_lst.append(batch["labels"][idx:idx+2].argmax(dim=0))
            idx += 2

        num_rows = int(len(outputs.logits)/2)
        probas = torch.cat(proba_lst).view(num_rows, 2)
        labels = torch.stack(label_lst)

        self.pred_metric_binary.update(probas, labels)
        self.uncert_metric.update(probas, labels)

    def on_test_epoch_end(self):
        uncert_score = self.uncert_metric.compute()
        pred_score = self.pred_metric.compute()
        self.log(f"{self.uncert_metric_name} {self.target_lang}", uncert_score, prog_bar=True)
        self.log(f"{self.pred_metric_name} {self.target_lang}", pred_score, prog_bar=True)
        self.result_lst.append(
            [self.exp_name,
             self.seed,
             self.target_lang, 
             self.uncert_metric_name,
             float(uncert_score)]
        )
        self.result_lst.append(
            [self.exp_name,
             self.seed,
             self.target_lang,
             self.pred_metric_name,
             float(pred_score)]
        )
        # reset metrics for the next target language
        self.uncert_metric.reset()
        self.pred_metric.reset()

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, self.optimizer, self.lr)
        # we don't use a scheduler for mad-x,
        # because the authors of the mad-x paper don't use one
        if self.task_adapter_name is None:
            scheduler = get_scheduler(
                "linear",
                optimizer, 
                num_warmup_steps=self.trainer.estimated_stepping_batches*0.1, 
                num_training_steps=self.trainer.estimated_stepping_batches)
            scheduler_dict = {"optimizer": optimizer,
                              "lr_scheduler": {
                                    "scheduler": scheduler,
                                    "interval": "step",
                                    "frequency": 1,
                                    "name": "lr_scheduler"}}
            return scheduler_dict
        else:
            return optimizer
        

def load_lightning_model(config, seed):
    if "copa" in config.trainer.exp_name:
        return LModelCopa(config, seed)
    else:
        return LModel(config, seed)