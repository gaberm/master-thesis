import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.model import load_model, load_tokenizer
from src.dataset import create_train_loader
from src.lightning import LModel

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    print(config)

    # trainer hyperparameters differ between systems
    system = platform.system()

    # load model, tokenizer and create data loaders
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    train_loader, val_loader = create_train_loader(config, tokenizer)
    print(train_loader)
    print(val_loader)
    
    # create lightning model and initialize wandb logger
    l_model = LModel(model, config)
    wandb_logger = WandbLogger(project=config.wandb.project, log_model="all", save_dir=config.data_dir[system])

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.data_dir[system]}/checkpoints/{config.trainer.exp_name}",
        monitor=config.params.val_metric,
        mode="max",
        filename=f"{{epoch}}-{{{config.params.val_metric}:.3f}}",
        save_top_k=config.trainer.save_top_k)

    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                        logger=wandb_logger, 
                        default_root_dir=config.data_dir[system],
                        deterministic=True,
                        strategy=config.trainer.strategy[system],
                        devices=config.trainer.devices[system],
                        callbacks=[checkpoint_callback])
    
    # train the model
    trainer.fit(model=l_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()