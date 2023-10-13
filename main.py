from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import evaluate
from tqdm.auto import tqdm
import lang2vec.lang2vec as l2v
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from optimizer import create_optimizer
from model import load_model
from dataset import load_dataset, tokenize_function, tokenize_dataset

class plAutoModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.metric = load_metric(cfg)
    
    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    def configure_optimizers(self):
        optimizer = create_optimizer(self, self.cfg)
        return optimizer

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    dataset = load_dataset(cfg)
    model, tokenizer = load_model(cfg)
    
    tokenized_dataset = tokenize_dataset(cfg, dataset, tokenizer)
    train_loader = tokenized_dataset
    
    wandb_logger = WandbLogger(project="master-thesis", log_model="all")
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=WandbLogger)

    # log gradients and model topology
    wandb_logger.watch(model)
    trainer.fit(model=model, train_dataloaders=train_loader)


# training settings
train_dataloader = dataloaders["en"]["train"] # finetuning the model in English
eval_dataloader = dataloaders["en"]["validation"]

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
#lr_scheduler = get_scheduler(
#    "linear",
#    optimizer=optimizer,
#    num_warmup_steps=0,
#    num_training_steps=num_training_steps,
#)
progress_bar = tqdm(range(num_training_steps))

metric = evaluate.load("f1")
best_val_score = 0
best_epoch = 0

# wandb
wandb.login()
wandb.init(project='Sprint 2')

## training the model

model.train() # changing model to training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        running_loss += loss.item()
        
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))
            wandb.log({'epoch': epoch+1, 'loss': running_loss/2000})
            running_loss = 0.0

    # validating the model after each epochs
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    val_score = metric.compute(average="micro")["f1"]
    print(f"F1 score epoch {epoch}: {val_score}")

    if val_score > best_val_score:
        checkpoint_path = f"bert_checkpoint_epoch_{epoch}.pt"
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        #torch.save(model.state_dict(), checkpoint_path)
        best_val_score = val_score
        best_epoch = epoch

print(f"The best model is from epoch: {best_epoch}.")