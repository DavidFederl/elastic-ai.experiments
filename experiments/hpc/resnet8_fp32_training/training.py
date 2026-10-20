import os
import random
from pathlib import Path
from typing import Any

from src.nn.data.cifar10 import CIFAR10_H5
from src.nn.model.resnet8 import FPModel as ResNet8
from src.nn.training import set_initial_seed
from src.utils.helper import setup_logging


def get_env_variables() -> dict[str, Any]:
    """Get environment variables from SLURM.

    Returns:
        dict[str, str]: Environment variables.
    """
    env_vars = {
        "DATASET_SRC": "",
        "LOG_DIR": Path().cwd(),
        "SEED": random.randint(0, 2**32 - 1),
    }
    for var in env_vars.keys():
        env_vars[var] = os.environ.get(var, env_vars[var])
    return env_vars


if __name__ == "__main__":
    vars = get_env_variables()

    setup_logging(log_dir=Path(vars["LOG_DIR"]), verbose=True)

    set_initial_seed(seed=int(vars["SEED"]), make_determenistic=True)
    model = ResNet8()
    training_set = CIFAR10_H5(
        path=vars["DATASET_SRC"],
        train=True,
        transform=None,
        target_transform=None,
    )
    validation_set = CIFAR10_H5(
        path=vars["DATASET_SRC"],
        train=False,
        transform=None,
        target_transform=None,
    )

    # TODO: Implement training loop here. For now, just print the model and dataset sizes.
    print(model)
    print(f"Training set size: {len(training_set)}")
    print(f"Validation set size: {len(validation_set)}")
