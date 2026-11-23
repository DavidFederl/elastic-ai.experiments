import logging

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 512,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    logger.info(
        f"Dataloader from {dataset.__class__.__name__} with {batch_size} samples per batch "
        f"({shuffle=}, {num_workers=}, {pin_memory=})"
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
