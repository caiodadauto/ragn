from functools import partial

import hydra
import numpy as np
from omegaconf import DictConfig

from ragn.ragn import RAGN
from ragn.data import init_generator


@hydra.main(config_path="./", config_name="ragn_config")
def my_app(cfg: DictConfig) -> None:
    model = model = RAGN(**cfg.model)
    batch_generator = partial(
        init_generator,
        scale_features=True,
        random_state=np.random.RandomState(123),
    )
    val_generator = batch_generator(cfg.data.val_data_path, 1)
    for in_graphs, _, path in val_generator:
        print()
        print("=========================================================================")
        print("NEW GRAPH from", path)
        _ = model(in_graphs, False)
        print("=========================================================================")


if __name__ == "__main__":
    my_app()
