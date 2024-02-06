import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model, load_tokenizer
from src.dataset import create_data_loaders
from src.lightning import LModel

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    print(config)

    # load model, tokenizer and create data loaders
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    train_loader, val_loader, _ = create_data_loaders(config, tokenizer)
    
    # create lightning model and initialize wandb logger
    l_model = LModel(model, config)

    wandb_dir = config.data_dir_mac if platform.system() == "Darwin" else config.data_dir_linux
    wandb_logger = WandbLogger(project=config.project, log_model="all", save_dir=wandb_dir)
    wandb_logger.watch(l_model)
    
    # create trainer depending on the OS
    # on the Linux server, we can use multiple GPUs
    if platform.system() == "Darwin":
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                            logger=wandb_logger, 
                            default_root_dir=config.data_dir_mac,
                            deterministic=True)
    else:
        trainer = pl.Trainer(max_epochs=config.params.max_epochs,
                            logger=wandb_logger, 
                            default_root_dir=config.data_dir_linux,
                            deterministic=True,
                            strategy="ddp",
                            devices=config.params.devices)
    
    # train the model
    trainer.fit(model=l_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()