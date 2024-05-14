import hydra
from omegaconf import DictConfig

from ragn.train import train_ragn


@hydra.main(config_path="./", config_name="ragn")
def my_app(cfg: DictConfig) -> None:
    train_ragn(
        cfg.mlflow.exp_name,
        cfg.mlflow.exp_tags,
        cfg.mlflow.run_tags,
        cfg.mlflow.run_id,
        cfg.mlflow.get_last_run,
        cfg.data,
        cfg.model,
        cfg.train,
    )


if __name__ == "__main__":
    my_app()

