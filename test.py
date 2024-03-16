import hydra
import dotenv
import platform
import os
import lightning.pytorch as pl
import pandas as pd
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from src.dataset import get_data_loader
from src.lightning import LModel
from src.utils import get_best_checkpoint, get_device, save_test_results, create_test_csv 

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    for seed in config.params.seeds:
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 
        
        # print
        print(config)

        # load model
        # model = load_model(config)
        ckpt_path = get_best_checkpoint(f"{config.data_dir[platform.system().lower()]}/checkpoints/{config.trainer.exp_name}/seed_{seed}")
        device = get_device(config)
        l_model = LModel.load_from_checkpoint(ckpt_path, map_location=device)
        l_model.model.eval()

        # create test data loaders
        # tokenizer = load_tokenizer(config)
        test_loaders = get_data_loader(config, "test")
        
        wandb_logger = WandbLogger(project=config.wandb.project, log_model="all")
        wandb_logger.watch(l_model)
        
        system = platform.system().lower()
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            logger=wandb_logger, 
            default_root_dir=config.data_dir[system],
            deterministic=True,
            strategy=config.trainer.strategy[system],
            devices=config.trainer.devices[system]
        )

        
        for lang, test_loader in zip(config.dataset.lang, test_loaders):
            l_model.target_lang = lang
            trainer.test(model=l_model, dataloaders=test_loader)
        
        save_test_results(l_model, config, seed)

    # create result csv
    create_test_csv(config.trainer.exp_name)

if __name__ == "__main__":
    main()