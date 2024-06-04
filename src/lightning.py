from lightning import LightningModule
from torch.optim import AdamW
import adapters
import torch
import os
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
        
        self.source_lang = config.params.source_lang
        self.target_lang = None
        if "madx" in config.keys():
            self.task_adapter_name = config.madx.task_adapter.name
            self.has_task_adapter = True
            if "lang_adapter" in config.madx.keys():
                self.has_lang_adapter = True
                self.type = "tala"
            else:
                self.has_lang_adapter = False
                self.type = "ta"
        else:
            self.task_adapter_name = None
            self.has_task_adapter = False
            self.has_lang_adapter = False
            self.type = "fft"
        
        if config.params.label_smoothing == 0.0:
            self.calibration = "none"
        else:
            self.calibration = "ls"
        
        self.lr = config.params.lr
        self.weight_decay = config.params.weight_decay
        self.warmup = config.params.warmup
        self.num_labels = config.model.num_labels
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=config.params.label_smoothing)
        # self.temperature = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        self.temperature = None
        self.data_dir = config.data_dir
        self.exp_name = config.trainer.exp_name
        self.task = config.dataset.name
        self.model_name = config.model.name
        self.ckpt_avg = ""
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
        if self.temperature is None:
            logits = outputs.logits
        else:
            logits = outputs.logits / self.temperature
        # logits = outputs.logits / self.temperature
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
             self.task,
             self.model_name,
             self.type,
             self.ckpt_avg,
             self.calibration,
             self.seed,
             self.target_lang, 
             self.uncert_metric_name,
             float(uncert_score)]
        )
        self.result_lst.append(
            [self.exp_name,
             self.task,
             self.model_name,
             self.type,
             self.ckpt_avg,
             self.calibration,
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
    
    def set_temperature(self, val_loader):
        self.calibration = "ts"
        self.temperature = torch.nn.Parameter(torch.tensor([1.5], device=self.device))

        if self.has_task_adapter:
            self.model.set_active_adapters(None)
            if self.has_lang_adapter:
                self.model.active_adapters = adapters.Stack(
                    self.source_lang,
                    self.task_adapter_name
                )
            else:
                self.model.active_adapters = self.task_adapter_name

        logits = []
        labels = []
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits.append(outputs.logits)
            labels.append(batch["labels"])
        
        nll_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = nll_loss(torch.cat(logits) / self.temperature, torch.cat(labels))
            loss.backward()
            return loss
        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.2f}")
    

class CopaModel(LightningModule):
    def __init__(self, config, seed):
        super().__init__()
        self.encoder, self.classifier = load_model(config)
        self.pred_metric_binary = load_metric(config, "pred", 2)
        self.pred_metric_multiclass = load_metric(config, "pred", 3)
        self.pred_metric_name = config.params.pred_metric
        self.uncert_metric = load_metric(config, "uncert")
        self.uncert_metric_name = config.params.uncert_metric

        self.source_lang = config.params.source_lang
        self.target_lang = None
        if "madx" in config.keys():
            self.task_adapter_name = config.madx.task_adapter.name
            self.has_task_adapter = True
            if "lang_adapter" in config.madx.keys():
                self.has_lang_adapter = True
                self.type = "tala"
            else:
                self.has_lang_adapter = False
                self.type = "ta"
        else:
            self.task_adapter_name = None
            self.has_task_adapter = False
            self.has_lang_adapter = False
            self.type = "fft"
        
        if config.params.label_smoothing == 0.0:
            self.calibration = "none"
        else:
            self.calibration = "ls"

        self.lr = config.params.lr
        self.weight_decay = config.params.weight_decay
        self.warmup = config.params.warmup
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=config.params.label_smoothing)
        self.temperature = None
        # self.temperature = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        self.data_dir = config.data_dir
        self.exp_name = config.trainer.exp_name
        self.task = config.dataset.name
        self.model_name = config.model.name
        self.ckpt_avg = ""
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
        clf_batches = {}
        for ds_name, ds_batch in batch.items():
            try:
                ds_batch = {k: v.to(self.device) for k, v in ds_batch.items()}
                outputs = self.encoder(
                    ds_batch["input_ids"],
                    ds_batch["attention_mask"],
                )
                clf_batches[ds_name] = {
                    "logits": self.classifier(outputs.pooler_output),
                    "labels": ds_batch["labels"],
                }
            # since copa has a smaller numbers of samples than siqa, we can run into an attribute error
            # when we try to access the copa batch which is already exhausted
            # we can ignore this error
            except AttributeError:
                pass

        loss = torch.tensor(0.0, device="cuda")
        for ds_name, batch in clf_batches.items():
            if ds_name == "copa":
                dim1 = int(len(batch["logits"]) / 2)
                new_logits = batch["logits"].view(dim1, 2)
                new_labels = batch["labels"].view(dim1, 2).argmax(dim=1)
            else:
                dim1 = int(len(batch["logits"]) / 3)
                new_logits = batch["logits"].view(dim1, 3)
                new_labels = batch["labels"].view(dim1, 3).argmax(dim=1)
            loss += self.ce_loss(new_logits, new_labels)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        clf_batches = {}
        for ds_name, ds_batch in batch.items():
            try:
                ds_batch = {k: v.to(self.device) for k, v in ds_batch.items()}
                outputs = self.encoder(
                    ds_batch["input_ids"],
                    ds_batch["attention_mask"],
                )
                clf_batches[ds_name] = {
                    "logits": self.classifier(outputs.pooler_output),
                    "labels": ds_batch["labels"],
                }
            except AttributeError:
                pass

        for ds_name, batch in clf_batches.items():
            if ds_name == "copa":
                dim1 = int(len(batch["logits"]) / 2)
                probs = batch["logits"].view(dim1, 2).softmax(dim=1)[:, 1]
                new_labels = batch["labels"].view(dim1, 2).argmax(dim=1)
                self.pred_metric_binary.update(probs, new_labels)
                self.copa_val_samples += len(new_labels)
            else:
                dim1 = int(len(batch["logits"]) / 3)
                probs = batch["logits"].view(dim1, 3).softmax(dim=1)
                new_labels = batch["labels"].view(dim1, 3).argmax(dim=1)
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
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.encoder(
                    batch["input_ids"],
                    batch["attention_mask"],
                )
        if self.temperature is None:
            logits = self.classifier(outputs.pooler_output)
        else:
            logits = self.classifier(outputs.pooler_output) / self.temperature
        # logits = outputs.logits / self.temperature
        dim_1 = int(len(logits) / 2)
        probs = logits.view(dim_1, 2).softmax(dim=1)[:, 1]
        new_labels = batch["labels"].view(dim_1, 2).argmax(dim=1)
        self.pred_metric_binary.update(probs, new_labels)
        self.uncert_metric.update(probs, new_labels)

    def on_test_epoch_end(self):
        uncert_score = self.uncert_metric.compute()
        pred_score = self.pred_metric_binary.compute()
        self.log(f"{self.uncert_metric_name} {self.target_lang}", uncert_score, prog_bar=True)
        self.log(f"{self.pred_metric_name} {self.target_lang}", pred_score, prog_bar=True)
        self.result_lst.append(
            [self.exp_name,
             self.task,
             self.model_name,
             self.task,
             self.ckpt_avg,
             self.calibration,
             self.seed,
             self.target_lang, 
             self.uncert_metric_name,
             float(uncert_score)]
        )
        self.result_lst.append(
            [self.exp_name,
             self.task,
             self.model_name,
             self.task,
             self.ckpt_avg,
             self.calibration,
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

    def set_temperature(self, val_loader):
        self.calibration = "ts"
        self.temperature = torch.nn.Parameter(torch.tensor([1.5], device=self.device))

        if self.has_task_adapter:
            self.encoder.set_active_adapters(None)
            if self.has_lang_adapter:
                self.encoder.active_adapters = adapters.Stack(
                    self.source_lang,
                    self.task_adapter_name
                )
            else:
                self.encoder.active_adapters = self.task_adapter_name
        
        logits = {"copa": [], "siqa": []}
        labels = {"copa": [], "siqa": []}
        for batch in val_loader:
            for ds_name in ["copa", "siqa"]:
                try:
                    ds_batch = {k: v.to(self.device) for k, v in batch[0][ds_name].items()}
                    with torch.no_grad():
                        enc_output = self.encoder(
                            ds_batch["input_ids"],
                            ds_batch["attention_mask"],
                        )
                        logits[ds_name].append(self.classifier(enc_output.pooler_output))
                        labels[ds_name].append(ds_batch["labels"])
                except AttributeError:
                    pass
        
        copa_logits = torch.cat(logits["copa"])
        dim_1 = int(copa_logits.size()[0] / 2)
        copa_logits = copa_logits.view(dim_1, 2)
        copa_labels = torch.cat(labels["copa"]).view(dim_1, 2).argmax(dim=1)
        siqa_logits = torch.cat(logits["siqa"])
        dim_1 = int(siqa_logits.size()[0] / 3)
        siqa_logits = siqa_logits.view(dim_1, 3)
        siqa_labels = torch.cat(labels["siqa"]).view(dim_1, 3).argmax(dim=1)
        
        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = ce_loss(copa_logits / self.temperature, copa_labels) + ce_loss(siqa_logits / self.temperature, siqa_labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.2f}")
    

def load_l_model(config, seed):
    load_copa_model = "copa" in config.trainer.exp_name
    # test: load model from checkpoint
    if config.model.load_ckpt:
        exp_dir = f"{config.data_dir}/checkpoints/{config.model.ckpt_dir}"
        test_dir = f"{exp_dir}/seed_{seed}"
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Directory {test_dir} not found. Do your seeds match the seeds used during training?")
        best_ckpt = find_best_ckpt(test_dir)
        device = get_device(config)
        
        # load lightning model using the best (most accurate) checkpoint 
        # based on the source language validation dataset
        if config.model.ckpt_avg == "none":
            if load_copa_model:
                l_model = CopaModel.load_from_checkpoint(best_ckpt, map_location=device, config=config)
                l_model.encoder.eval()
                l_model.classifier.eval()
            else:
                l_model = LModel.load_from_checkpoint(best_ckpt, map_location=device, config=config)
                l_model.model.eval()
            l_model.exp_name = config.trainer.exp_name
            l_model.ckpt_avg = "none"
            return l_model
        
        # checkpoint averaging: replace the self.model's state_dict with the averaged state_dict
        else:
            if load_copa_model:
                l_model = CopaModel.load_from_checkpoint(best_ckpt, map_location=device, config=config)
                enc_state_dict, cls_state_dict = compute_ckpt_average(test_dir, device, config.model.ckpt_avg)
                l_model.encoder.load_state_dict(enc_state_dict)
                l_model.classifier.load_state_dict(cls_state_dict)
                l_model.encoder.eval()
                l_model.classifier.eval()
            else:
                l_model = LModel.load_from_checkpoint(best_ckpt, map_location=device, config=config)
                state_dict = compute_ckpt_average(test_dir, device, config.model.ckpt_avg)
                l_model.model.load_state_dict(state_dict)
                l_model.model.eval()
            l_model.exp_name = config.trainer.exp_name
            l_model.ckpt_avg = config.model.ckpt_avg
            return l_model 
    
    # train: load model from scratch
    else:
        if load_copa_model:
            return CopaModel(config, seed)
        else:
            return LModel(config, seed)