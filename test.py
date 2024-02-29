import hydra
import dotenv
import platform
import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.loggers import WandbLogger
from src.model import load_model, load_tokenizer
from src.dataset import create_test_loader
from src.lightning import LModel
from src.utils import get_best_checkpoint, get_device

dotenv.load_dotenv(override=True)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    # seed everything for reproducibility
    pl.seed_everything(config.params.seed, workers=True) 
    
    print(config)

    # load model
    model = load_model(config)
    ckpt_path = get_best_checkpoint(config.model.ckpt_dir)
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
    result_df = pd.DataFrame(l_model.result_lst, columns=["target_lang", "metric", "score"])
    result_df.to_csv(f"res/test/{config.trainer.exp_name}", index=False)

if __name__ == "__main__":
    main()