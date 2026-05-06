import logging
from pathlib import Path
from typing import Callable

import torch
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )

        logger.info(
            f"Dataset: {self.__class__.__name__} ({'Training' if self.train else 'Test'})"
        )
        logger.debug(f"Total: {len(self.data)} samples")
        logger.debug(f"Classes: {list(enumerate(self.classes))}")
        logger.debug(
            f"Element Shape: {self.data[0].shape}={self.data[0].numel()} values total"
        )


def fashionmnist_trainingset_flattened() -> FashionMNIST:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    return FashionMNIST(root="datasets", train=True, transform=transformations)


def fashionmnist_validationset_flattened() -> FashionMNIST:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    return FashionMNIST(root="datasets", train=False, transform=transformations)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )
    ds = fashionmnist_trainingset_flattened()
    print(f"data={ds[0][0]}\ntarget={ds[1][0]}")
