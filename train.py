import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.model import load_model, load_tokenizer
from src.dataset import create_data_loaders
from src.lightning import LModel

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    print(config)

    sys = platform.system()

    # load model, tokenizer and create data loaders
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    train_loader, val_loader = create_data_loaders(config, tokenizer)
    
    # create lightning model and initialize wandb logger
    l_model = LModel(model, config)
    wandb_logger = WandbLogger(project=config.project, log_model="all", save_dir=config.data_dir[sys])

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.data_dir[sys]}/checkpoints/{config.trainer.exp_name}",
        monitor=config.params.val_metric,
        mode="max",
        filename=f"{{epoch}}-{{step}}-{{{config.params.val_metric}:.3f}}",
        save_top_k=config.trainer.save_top_k)

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                        logger=wandb_logger, 
                        default_root_dir=config.data_dir[sys],
                        deterministic=True,
                        strategy=config.trainer.strategy[sys],
                        devices=config.trainer.devices[sys],
                        callbacks=[checkpoint_callback])
    
    # train the model
    trainer.fit(model=l_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()