import hydra
import dotenv
import platform
import os
import lightning.pytorch as pl
import pandas as pd
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model, load_tokenizer
from src.dataset import create_test_loader
from src.lightning import LModel
from src.utils import get_best_checkpoint, get_device, create_test_csv

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    for seed in config.params.seeds:
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 
        
        # print
        print(config)

        # load model
        model = load_model(config)
        ckpt_path = get_best_checkpoint(f"{config.data_dir[platform.system().lower()]}checkpoints/{config.trainer.exp_name}/seed_{seed}")
        device = get_device(config)
        l_model = LModel.load_from_checkpoint(ckpt_path, model=model, map_location=device)
        l_model.model.eval()

        # create test data loaders
        tokenizer = load_tokenizer(config)
        test_loader = create_test_loader(config, tokenizer)
        
        wandb_logger = WandbLogger(project=config.wandb.project, log_model="all")
        wandb_logger.watch(l_model)
        
        system = platform.system().lower()
        trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                            logger=wandb_logger, 
                            default_root_dir=config.data_dir[system],
                            deterministic=True,
                            strategy=config.trainer.strategy[system],
                            devices=config.trainer.devices[system])

        for target_lang, test_loader in test_loader.items():
            l_model.target_lang = target_lang
            trainer.test(model=l_model, dataloaders=test_loader)
        
        # save test results as csv
        try:
            os.mkdir(f"res/test/{config.trainer.exp_name}")
        except FileExistsError:
            pass
        np.save(f"res/test/{config.trainer.exp_name}/seed_{seed}", l_model.result_lst)

    # create result csv
    create_test_csv(f"res/test/{config.trainer.exp_name}")

if __name__ == "__main__":
    main()