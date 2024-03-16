import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.dataset import get_data_loader
from src.lightning import load_lightning_model

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    # run the experiment for 5 different seeds
    for seed in config.params.seeds:
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 

        # print the config for the current run
        print(config)

        # gpu, no of devices and training strategy differ between the macos and linux setup
        # so we need to know which system we are on
        system = platform.system().lower()

        train_loader = get_data_loader(config, "train")
        val_loader = get_data_loader(config, "val")
        
        # create lightning model and initialize wandb logger
        l_model = load_lightning_model(config, seed)
        wandb_logger = WandbLogger(project=config.wandb.project)

        ckpt_dir = f"{config.data_dir[system]}/checkpoints/{config.trainer.exp_name}/seed_{seed}"
        # check to avoid overwriting of existing checkpoints
        # if os.path.exists(ckpt_dir):
        #     raise ValueError(f"Checkpoint directory {ckpt_dir} already exists. Please delete or change it.")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=config.params.pred_metric,
            mode="max",
            filename=f"{{epoch}}-{{step}}-{{{config.params.pred_metric}:.3f}}",
            save_top_k=config.trainer.save_top_k
        )
        lr_callback = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            logger=wandb_logger, 
            default_root_dir=config.data_dir[system],
            deterministic=True,
            strategy=config.trainer.strategy[system],
            devices=config.trainer.devices[system],
            val_check_interval=0.25,
            callbacks=[checkpoint_callback, lr_callback]
        )
        
        # train the model
        trainer.fit(
            model=l_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

if __name__ == "__main__":
    main()