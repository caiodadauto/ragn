import hydra
from omegaconf import DictConfig

from ragn.train import train_ragn


@hydra.main(config_path="./", config_name="ragn_config")
def my_app(cfg: DictConfig) -> None:
    train_ragn(
        cfg.mlflow.exp_name,
        cfg.mlflow.load_runs,
        cfg.data,
        cfg.model,
        cfg.train,
        cfg.num_msg,
        cfg.seed,
        cfg.debug,
    )


if __name__ == "__main__":
    my_app()
