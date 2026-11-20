import logging
from pathlib import Path
from typing import Callable

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10(torchvision.datasets.CIFAR10):
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
            f"Element Shape: {self.data[0].shape}={self.data[0].size} values total"
        )


def cifar10_trainingset() -> CIFAR10:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return CIFAR10(root="datasets", train=True, transform=transformations)


def cifar10_validationset() -> CIFAR10:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return CIFAR10(root="datasets", train=False, transform=transformations)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )
    ds = cifar10_trainingset()
    print(f"data={ds[0][0]}\ntarget={ds[0][1]}")
