import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    training_loader: DataLoader
    validation_loader: DataLoader
    classes: dict[str, int]
    element_shape: torch.Size
    batch_size: int
    name: str

    def __init__(self) -> None:
        self._print_dataset_info()

    def _print_dataset_info(self) -> None:
        def print_classes(classes: dict[str, int]):
            classes_str: str = "Classes: {"
            for class_name, class_id in classes.items():
                classes_str += "  {}: '{}'".format(class_id, class_name)
            classes_str += "}"
            return classes_str

        logger.info(f"Dataset: {self.name}")
        logger.debug(print_classes(self.classes))
        logger.debug(f"Element Size: {self.element_shape}={self.element_shape.numel()}")
        logger.debug(f"Batch Size: {self.batch_size}")
        logger.debug(
            f"Total: {len(self.training_loader) + len(self.validation_loader)} batches"
        )
        logger.debug(f"Training: {len(self.training_loader) * self.batch_size} samples")
        logger.debug(
            f"Validation: {len(self.validation_loader) * self.batch_size} samples"
        )
