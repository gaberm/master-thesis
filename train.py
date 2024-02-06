import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model
from src.dataset import create_data_loaders
from src.lightning import LModel

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    print(config)

    # load model and create data loaders
    model, tokenizer = load_model(config)
    train_loader, val_loader, _ = create_data_loaders(config, tokenizer)
    
    # create lightning model and initialize wandb logger
    pl_model = LModel(model, config)

    if platform.system() == "Darwin":
        wandb_dir = config.output_dir_mac
    else:
        wandb_dir = config.output_dir_linux
    wandb_logger = WandbLogger(project=config.project, log_model="all", save_dir=wandb_dir)
    wandb_logger.watch(pl_model)
    
    # create trainer depending on the OS
    # on the Linux server, we can use multiple GPUs
    if platform.system() == "Darwin":
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                            logger=wandb_logger, 
                            default_root_dir=config.output_dir_mac,
                            deterministic=True)
    else:
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                            logger=wandb_logger, 
                            default_root_dir=config.output_dir_linux,
                            deterministic=True,
                            strategy="ddp",
                            devices=config.devices)
    
    # train the model
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()