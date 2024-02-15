import hydra
import dotenv
import platform
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model, load_tokenizer
from src.dataset import create_data_loaders
from src.lightning import LModel

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    print(config)

    # load model
    l_model = LModel.load_from_checkpoint(config.model.ckpt_path)

    # create test data loaders
    tokenizer = load_tokenizer(config)
    test_loaders = create_data_loaders(config, tokenizer)
    
    wandb_logger = WandbLogger(project=config.project, log_model="all")
    wandb_logger.watch(l_model)
    
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

    for target_lang, test_loader in test_loaders.items():
        l_model.target_lang = target_lang
        trainer.test(model=l_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()