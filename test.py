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
    ckpt_path = config.model.ckpt_path
    model = load_model(config)
    l_model = LModel.load_from_checkpoint(ckpt_path, model=model)

    # create test data loaders
    tokenizer = load_tokenizer(config)
    test_loaders = create_data_loaders(config, tokenizer)
    
    wandb_logger = WandbLogger(project=config.wandb.project, log_model="all")
    wandb_logger.watch(l_model)
    
    system = platform.system()
    trainer = pl.Trainer(max_epochs=config.trainer.max_epochs,
                        logger=wandb_logger, 
                        default_root_dir=config.data_dir[system],
                        deterministic=True,
                        strategy=config.trainer.strategy[system],
                        devices=config.trainer.devices[system])

    for target_lang, test_loader in test_loaders.items():
        l_model.target_lang = target_lang
        trainer.test(model=l_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()