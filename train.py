import hydra
import dotenv
import os
import shutil
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from src.dataset import get_data_loader
from src.lightning import load_l_model

dotenv.load_dotenv(".env")

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(config):
    # run the experiment for 5 different seeds
    for seed in [0, 1, 2, 3, 4]:
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 

        # print the config for the current run
        print(config)

        train_loader = get_data_loader(config, "train")
        val_loader = get_data_loader(config, "validation")
        
        # create lightning model and initialize wandb logger
        l_model = load_l_model(config, seed)
        wandb_logger = WandbLogger(project=config.wandb.project)

        # skip training if checkpoint directory already exists
        # this is useful since many experiments share the same checkpoints
        ckpt_dir = f"{config.data_dir}/checkpoints/{config.model.ckpt_dir}/seed_{seed}"
        if os.path.exists(ckpt_dir):
            if config.params.overwrite_ckpt:
                shutil.rmtree(ckpt_dir)
                print(f"Removed existing checkpoint directory {ckpt_dir} for experiment {config.trainer.exp_name}")
            else:
                print(f"Checkpoint directory {ckpt_dir} already exists for experiment {config.trainer.exp_name}. Skipping training for seed {seed}. Use ++params.overwrite_ckpt=True to overwrite existing checkpoints.")
                continue

        lr_callback = LearningRateMonitor(logging_interval="step")
        val_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=config.params.pred_metric,
            mode="max",
            filename=f"{{epoch}}-{{step}}-{{{config.params.pred_metric}:.3f}}",
            save_top_k=config.trainer.save_top_k,
        )

        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            logger=wandb_logger, 
            default_root_dir=config.data_dir,
            deterministic=True,
            strategy=config.trainer.strategy,
            devices=config.trainer.devices,
            val_check_interval=0.25,
            callbacks=[lr_callback, val_checkpoint],
        )

        # train the model
        trainer.fit(
            model=l_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
if __name__ == "__main__":
    main()
    