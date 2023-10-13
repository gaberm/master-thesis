from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    for i in cfg.dataset.values():
        print(i.languages)
    #ds = [dataset.languages for dataset in cfg.dataset]
    #print(ds)

if __name__ == "__main__":
    my_app()