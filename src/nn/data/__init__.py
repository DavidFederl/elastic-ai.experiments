from .cifar10 import CIFAR10, cifar10_trainingset, cifar10_validationset
from .fashionmnist import (
    FashionMNIST,
    fashionmnist_trainingset_flattened,
    fashionmnist_validationset_flattened,
)
from .smartable import (
    SmarTable,
    smartable_trainingset_flattened,
    smartable_validationset_flattened,
)
from .utils import get_dataloader

__all__ = [
    "CIFAR10",
    "cifar10_trainingset",
    "cifar10_validationset",
    "FashionMNIST",
    "fashionmnist_trainingset_flattened",
    "fashionmnist_validationset_flattened",
    "SmarTable",
    "smartable_trainingset_flattened",
    "smartable_validationset_flattened",
    "get_dataloader",
]
