import hydra
import dotenv
import os
import shutil
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.dataset import get_data_loader
from src.lightning import get_l_model

dotenv.load_dotenv(".env")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    # run the experiment for 5 different seeds
    for idx, seed in enumerate(config.params.seeds):
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 

        # print the config for the current run
        print(config)

        train_loader = get_data_loader(config, "train")
        val_loader = get_data_loader(config, "validation")
        
        # create lightning model and initialize wandb logger
        l_model = get_l_model(config, seed, idx)
        wandb_logger = WandbLogger(project=config.wandb.project)

        ckpt_dir = f"{config.data_dir}/checkpoints/{config.trainer.exp_name}/seed_{seed}"
        # check to avoid overwriting of existing checkpoints
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            print(f"Removed existing checkpoint directory {ckpt_dir} for experiment {config.trainer.exp_name}")
        
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
            default_root_dir=config.data_dir,
            deterministic=True,
            strategy=config.trainer.strategy,
            devices=config.trainer.devices,
            val_check_interval=0.25,
            callbacks=[checkpoint_callback, lr_callback],
            use_distributed_sampler=False
        )

        # train the model
        trainer.fit(
            model=l_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
if __name__ == "__main__":
    main()
    