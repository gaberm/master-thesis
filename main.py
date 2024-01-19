import hydra
import dotenv
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model
from src.dataset import create_data_loaders
from src.lightning import LModel

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig):
    print(config)
    model, tokenizer, adapter_names = load_model(config)
    train_loader, val_loader, test_loaders = create_data_loaders(config, tokenizer)
    
    pl_model = LModel(model, config, adapter_names)
    wandb_logger = WandbLogger(project=config.project, log_model="all")
    wandb_logger.watch(pl_model)
    
    if config.model.load_pretrained:
        pass # TODO: load pretrained model
    else:
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                             logger=wandb_logger, 
                             default_root_dir=config.checkpoint_dir,
                             deterministic=True)
        trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    for target_lang, test_loader in test_loaders.items():
        pl_model.target_lang = target_lang
        trainer.test(model=pl_model, test_dataloaders=test_loader)

if __name__ == "__main__":
    main()