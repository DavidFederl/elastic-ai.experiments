import logging
from dataclasses import dataclass

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    training_loader: DataLoader
    validation_loader: DataLoader
    classes: dict[str, int]
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

        logger.info("Dataset: {}".format(self.name))
        logger.debug(print_classes(self.classes))
        logger.debug(
            "Total: {} samples".format(
                len(self.training_loader) + len(self.validation_loader)
            )
        )
        logger.debug("Training: {} samples".format(len(self.training_loader)))
        logger.debug("Validation: {} samples".format(len(self.validation_loader)))
