from pyparsing import Any
from transformers.adapters.composition import Stack
import torch
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from optimizer import create_optimizer
from model import load_model
from dataset import create_data_loaders
from metric import load_metric
from dotenv import load_dotenv

load_dotenv()

class LightningModel(pl.LightningModule):
    def __init__(self, model: Any, config: DictConfig):
        super().__init__()
        self.model = model
        self.metric = load_metric(config)
        self.metric_name = config.params.metric
        self.config = config
    
    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
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

    def test_step(self, batch, batch_idx, dataloader_idx):
        outputs = self.model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=batch["labels"])
        return outputs
    
    def on_test_epoch_start(self):
        self.model.active_adapters = Stack(
            self.config.model.adapter.task_adapter_args.checkpoint,
            self.target_languages
        )

    def on_test_epoch_end(self):
        val_score = self.metric.compute()[self.metric_name]
        self.log(f'val_{self.metric_name}', val_score)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        return optimizer

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig):

    model, tokenizer = load_model(config)
    train_loader, val_loader, test_loader = create_data_loaders(config, tokenizer)
    
    datasets = [config.train_datasets] + [config.val_datasets] + [config.test_datasets]

    for dataset in config.datasets:
        set = load_datasets(dataset)
        tokenized_set = tokenize_for_crosslingual_transfer(config, set, tokenizer)
        if set in config.train_datasets:
            train_loader.append(tokenized_set)
        elif set in config.val_datasets:
            val_loader.append(tokenized_set)
        else:
            test_loader.append(tokenized_set)
    
    pl_model = LightningModel(model, config)
    wandb_logger = WandbLogger(project=config.project, log_model="all")
    wandb_logger.watch(pl_model)
    
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    
    model, tokenizer = load_model(config)
    tokenized_datasets = tokenize_for_crosslingual_transfer(config, datasets, tokenizer)

    train_loader = [loader_container.loader for loader_container in tokenized_datasets 
                    if loader_container.name == "xnli" 
                    and loader_container.split == "train"][0]
    val_loader = [loader_container.loader for loader_container in tokenized_datasets 
                    if loader_container.name == "xnli" 
                    and loader_container.split == "validation" 
                    and loader_container.language == config.params.source_lang][0]
    test_loader = [loader_container.loader for loader_container in tokenized_datasets 
                    if loader_container.name == "xnli" 
                    and loader_container.split == "test"]

    pl_model = LightningModel(model, config)
    wandb_logger = WandbLogger(project="master-thesis", log_model="all")
    wandb_logger.watch(pl_model)
    

    trainer.test(model=pl_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()

    # xnli_model.target_lang = xnli_target_lang
    # xcopa_model.target_lang = xcopa_target_lang
    
    # xcopa_trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger)
        
    # xcopa_model = LightningModel(model, config)

    # # tokenize datasets and create dataloaders
    # tokenized_datasets = tokenize_for_crosslingual_transfer(config, datasets, tokenizer)
    
    # xnli_target_lang = [loader.language for loader in tokenized_datasets if loader.name == "xnli" and loader.split == "test"]
    # xcopa_train_loader = [loader for loader in tokenized_datasets if loader.name == "xcopa" and loader.split == "train"]
    # xcopa_val_loader = [loader for loader in tokenized_datasets if loader.name == "xcopa" and loader.split == "validation"]
    # xcopa_test_loader = [loader for loader in tokenized_datasets if loader.name == "xcopa" and loader.split == "test"]
    # xcopa_target_lang = [loader.language for loader in tokenized_datasets if loader.name == "xcopa" and loader.split == "test"]
    
    

    # # log gradients and model topology
    
    # # train and test
    
    # xcopa_trainer.fit(model=xcopa_model, train_dataloaders=xcopa_train_loader, val_dataloaders=xcopa_val_loader)
    # xcopa_trainer.test(model=xcopa_model, dataloaders=xcopa_test_loader)








# # training settings
# train_dataloader = dataloaders["en"]["train"] # finetuning the model in English
# eval_dataloader = dataloaders["en"]["validation"]

# optimizer = AdamW(model.parameters(), lr=5e-5)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# num_epochs = 2
# num_training_steps = num_epochs * len(train_dataloader)
# #lr_scheduler = get_scheduler(
# #    "linear",
# #    optimizer=optimizer,
# #    num_warmup_steps=0,
# #    num_training_steps=num_training_steps,
# #)
# progress_bar = tqdm(range(num_training_steps))

# metric = evaluate.load("f1")
# best_val_score = 0
# best_epoch = 0

# # wandb
# wandb.login()
# wandb.init(project='Sprint 2')

# ## training the model

# model.train() # changing model to training mode
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for batch_idx, batch in enumerate(train_dataloader):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         #lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#         running_loss += loss.item()
        
#         if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, batch_idx + 1, running_loss / 2000))
#             wandb.log({'epoch': epoch+1, 'loss': running_loss/2000})
#             running_loss = 0.0

#     # validating the model after each epochs
#     for batch in eval_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)

#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)
#         metric.add_batch(predictions=predictions, references=batch["labels"])

#     val_score = metric.compute(average="micro")["f1"]
#     print(f"F1 score epoch {epoch}: {val_score}")

#     if val_score > best_val_score:
#         checkpoint_path = f"bert_checkpoint_epoch_{epoch}.pt"
#         model.save_pretrained(save_directory)
#         tokenizer.save_pretrained(save_directory)
#         #torch.save(model.state_dict(), checkpoint_path)
#         best_val_score = val_score
#         best_epoch = epoch

        

# print(f"The best model is from epoch: {best_epoch}.")
