from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyparsing import Any
from transformers.adapters.composition import Stack
import torch
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from src.optimizer import create_optimizer
from src.model import load_model
from src.dataset import create_data_loaders
from src.metric import load_metric
from dotenv import load_dotenv

load_dotenv()

class LightningModel(pl.LightningModule):
    def __init__(self, model: Any, config: DictConfig):
        super().__init__()
        self.model = model
        self.metric = load_metric(config)
        self.metric_name = config.params.metric
        self.lang_adapters = self.set_lang_adapter(config)
        self.task_adapter = self.set_task_adapter(config)
        self.source_lang = config.params.source_lang
        target_langs = ""
    
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
    def activate_adapters(self, tgt_lang: str):
        self.model.set_active_adapters(None)
        if self.task_adapter != None and self.lang_adapters != None:
            self.model.set_active_adapters(Stack(self.task_adapter, self.lang_adapters[tgt_lang]))
        elif self.lang_adapters != None:
            self.model.set_active_adapters(self.lang_adapters[tgt_lang])
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
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])
    
    def on_validation_epoch_end(self):
        val_score = self.metric.compute(average="micro")[self.metric_name]
        self.log(f"{self.metric_name}", val_score)
    
    def test_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])

    def on_test_epoch_end(self):
        val_score = self.metric.compute()[self.metric_name]
        self.log(f'{self.metric_name}_{tgt_lang}', val_score)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        return optimizer


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig):

    model, tokenizer = load_model(config)
    train_loader, val_loader, test_loaders = create_data_loaders(config, tokenizer)
    
    pl_model = LightningModel(model, config)
    wandb_logger = WandbLogger(project=config.project, log_model="all")
    wandb_logger.watch(pl_model)
    
    if config.load_pretrained:
        pass # TODO: load pretrained model
    else:
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                             logger=wandb_logger, 
                             default_root_dir=config.checkpoint_dir,
                             deterministic=True)
        trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    for tgt_lang, test_loader in test_loaders.items():
        pl_model.activate_adapters(tgt_lang)
        trainer.test(model=pl_model, test_dataloaders=test_loader)

if __name__ == "__main__":
    main()