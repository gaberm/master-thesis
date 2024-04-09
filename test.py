import hydra
import dotenv
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.dataset import get_data_loader
from src.lightning import load_l_model
from src.utils import save_test_results, create_result_csv 

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    for seed in config.params.seeds:
        # seed everything for reproducibility
        pl.seed_everything(seed, workers=True) 
        
        print(config)

        l_model = load_l_model(config, seed)

        # create test data loaders
        test_loaders = get_data_loader(config, "test")
        
        wandb_logger = WandbLogger(project=config.wandb.project, log_model="all")
        wandb_logger.watch(l_model)
        
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            logger=wandb_logger, 
            default_root_dir=config.data_dir,
            deterministic=True,
            strategy=config.trainer.strategy,
            devices=config.trainer.devices
        )
        
        for lang, test_loader in zip(config.dataset.test_lang, test_loaders):
            l_model.target_lang = lang
            trainer.test(model=l_model, dataloaders=test_loader)
        save_test_results(l_model, config, seed)

    create_result_csv(config.trainer.exp_name)

if __name__ == "__main__":
    main()