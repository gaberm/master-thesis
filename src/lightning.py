from lightning import LightningModule
from torch.optim import AdamW
import adapters
import torch
import os
import glob
from transformers import get_scheduler
from .utils import get_device, find_best_ckpt, compute_ckpt_average
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
        
        self.target_lang = None
        if "madx" in config.keys():
            self.task_adapter_name = config.madx.task_adapter.name
            self.has_task_adapter = True
            if "lang_adapter" in config.madx.keys():
                self.has_lang_adapter = True
            else:
                self.has_lang_adapter = False
        else:
            self.task_adapter_name = None
            self.has_task_adapter = False
            self.has_lang_adapter = False
        
        self.lr = config.params.lr
        self.weight_decay = config.params.weight_decay
        self.warmup = config.params.warmup
        self.num_labels = config.model.num_labels
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.data_dir = config.data_dir
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
        if self.has_task_adapter:
            self.model.set_active_adapters(None)
            if self.has_lang_adapter:
                self.model.active_adapters = adapters.Stack(
                    self.target_lang,
                    self.task_adapter_name
                )
            else:
                self.model.active_adapters = self.task_adapter_name
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        preds = outputs.logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        if self.num_labels == 2:
            probs = probs[:, 1]
        self.uncert_metric.update(probs, batch["labels"])
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
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # we don't use a scheduler for mad-x,
        # because the authors of the mad-x paper don't use one
        if self.task_adapter_name is None:
            scheduler = get_scheduler(
                "linear",
                optimizer, 
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.warmup, 
                num_training_steps=self.trainer.estimated_stepping_batches
            )
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
        

class LCopaModel(LightningModule):
    def __init__(self, config, seed):
        super().__init__()
        self.encoder, self.classifier = load_model(config)
        self.pred_metric_binary = load_metric(config, "pred", 2)
        self.pred_metric_multiclass = load_metric(config, "pred", 3)
        self.pred_metric_name = config.params.pred_metric
        self.uncert_metric = load_metric(config, "uncert")
        self.uncert_metric_name = config.params.uncert_metric

        self.target_lang = None
        if "madx" in config.keys():
            self.task_adapter_name = config.madx.task_adapter.name
            self.has_task_adapter = True
            if "lang_adapter" in config.madx.keys():
                self.has_lang_adapter = True
            else:
                self.has_lang_adapter = False
        else:
            self.task_adapter_name = None
            self.has_task_adapter = False
            self.has_lang_adapter = False

        self.lr = config.params.lr
        self.weight_decay = config.params.weight_decay
        self.warmup = config.params.warmup
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.data_dir = config.data_dir
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
        inputs = {}
        for dataset, data in batch.items():
            try:
                inputs[dataset] = {k: v.to(self.device) for k, v in data.items()}
            except AttributeError:
                pass

        classifier_logits = {}
        for dataset, batch in inputs.items():
            outputs = self.encoder(
                batch["input_ids"],
                batch["attention_mask"],
            )
            classifier_logits[dataset] = self.classifier(outputs.pooler_output)

        loss = torch.tensor(0.0, device="cuda")
        for dataset, logits in classifier_logits.items():
            if dataset == "copa":
                rows = int(len(logits) / 2)
                new_logits = logits.view(rows, 2)
                new_labels = inputs[dataset]["labels"].view(rows, 2).argmax(dim=1)
            if dataset == "siqa":
                rows = int(len(logits) / 3)
                new_logits = logits.view(rows, 3)
                new_labels = inputs[dataset]["labels"].view(rows, 3).argmax(dim=1)
            loss += self.ce_loss(new_logits, new_labels)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = {}
        for dataset, data in batch.items():
            try:
                inputs[dataset] = {k: v.to(self.device) for k, v in data.items()}
            except AttributeError:
                pass

        classifier_logits = {}
        for dataset, batch in inputs.items():
            outputs = self.encoder(
                batch["input_ids"],
                batch["attention_mask"],
            )
            classifier_logits[dataset] = self.classifier(outputs.pooler_output)

        for dataset, logits in classifier_logits.items():
            if dataset == "copa":
                rows = int(len(logits) / 2)
                probs = logits.view(rows, 2).softmax(dim=1)[:, 1]
                new_labels = inputs[dataset]["labels"].view(rows, 2).argmax(dim=1)
                self.pred_metric_binary.update(probs, new_labels)
                self.copa_val_samples += len(new_labels)
            if dataset == "siqa":
                rows = int(len(logits) / 3)
                probs = logits.view(rows, 3).softmax(dim=1)
                new_labels = inputs[dataset]["labels"].view(rows, 3).argmax(dim=1)
                self.pred_metric_multiclass.update(probs, new_labels)
                self.siqa_val_samples += len(new_labels)
    
    def on_validation_epoch_end(self):
        total_samples = self.copa_val_samples + self.siqa_val_samples
        val_score = (
            self.pred_metric_binary.compute() * self.copa_val_samples + 
            self.pred_metric_multiclass.compute() * self.siqa_val_samples
        ) / total_samples
        self.log(f"{self.pred_metric_name}", val_score, prog_bar=True, sync_dist=True)
        self.pred_metric_binary.reset()
        self.pred_metric_multiclass.reset()
        self.copa_val_samples = 0
        self.siqa_val_samples = 0

    def on_test_epoch_start(self):
        if self.has_task_adapter:
            self.encoder.set_active_adapters(None)
            if self.has_lang_adapter:
                self.encoder.active_adapters = adapters.Stack(
                    self.target_lang,
                    self.task_adapter_name
                )
            else:
                self.encoder.active_adapters = self.task_adapter_name
    
    def test_step(self, batch, batch_idx):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.encoder(
                batch["input_ids"],
                batch["attention_mask"],
            )
        logits = self.classifier(outputs.pooler_output)
        rows = int(len(logits) / 2)
        probs = logits.view(rows, 2).softmax(dim=1)[:, 1]
        new_labels = inputs["labels"].view(rows, 2).argmax(dim=1)
        self.pred_metric_binary.update(probs, new_labels)
        self.uncert_metric.update(probs, new_labels)

    def on_test_epoch_end(self):
        uncert_score = self.uncert_metric.compute()
        pred_score = self.pred_metric_binary.compute()
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
        self.pred_metric_binary.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # we don't use a scheduler for mad-x,
        # because the authors of the mad-x paper don't use one
        if self.task_adapter_name is None:
            scheduler = get_scheduler(
                "linear",
                optimizer, 
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.warmup, 
                num_training_steps=self.trainer.estimated_stepping_batches
            )
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
    

def load_l_model(config, seed):
    load_copa_model = "copa" in config.trainer.exp_name

    # test: load model from checkpoint
    if config.model.load_ckpt:
        if config.model.ckpt_averaging:
            exp = config.trainer.exp_name.replace("_ca", "")
        else:
            exp = config.trainer.exp_name
        exp_dir = f"{config.data_dir}/checkpoints/{exp}"
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Checkpoint directory {exp_dir} not found. Please train the model for {config.trainer.exp_name.replace("_ca", "")}.")
        run_dir = f"{exp_dir}/seed_{seed}"
        if not os.path.exists(run_dir):
            existing_seeds = [
                int(os.path.basename(dir).replace("seed_", "")) 
                for dir in glob.glob(f"{exp_dir}/seed_*")
            ]
            raise FileNotFoundError(f"Directory {run_dir} not found. Please use the following seeds: {existing_seeds}.")

        best_ckpt = find_best_ckpt(run_dir)
        device = get_device(config)
        # load lightning model using the best (most accurate) checkpoint 
        # based on the source language validation dataset
        if config.model.ckpt_averaging == "none":
            if load_copa_model:
                l_model = LCopaModel.load_from_checkpoint(best_ckpt, map_location=device)
                l_model.encoder.eval()
                l_model.classifier.eval()
            else:
                l_model = LModel.load_from_checkpoint(best_ckpt, map_location=device)
                l_model.model.eval()
            return l_model
        # checkpoint averaging: replace the self.model's state_dict with the averaged state_dict
        else:
            if load_copa_model:
                l_model = LCopaModel.load_from_checkpoint(best_ckpt, map_location=device)
                l_model.exp_name = config.trainer.exp_name
                enc_state_dict, cls_state_dict = compute_ckpt_average(run_dir, device, config.model.ckpt_averaging)
                l_model.encoder.load_state_dict(enc_state_dict)
                l_model.classifier.load_state_dict(cls_state_dict)
                l_model.encoder.eval()
                l_model.classifier.eval()
                return l_model
            else:
                l_model = LModel.load_from_checkpoint(best_ckpt, map_location=device)
                l_model.exp_name = config.trainer.exp_name
                state_dict = compute_ckpt_average(run_dir, device, config.model.ckpt_averaging)
                l_model.model.load_state_dict(state_dict)
                l_model.model.eval()
            return l_model 
            
    
    # train: load model from scratch
    else:
        if load_copa_model:
            return LCopaModel(config, seed)
        else:
            return LModel(config, seed)