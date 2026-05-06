import logging
from pathlib import Path
from typing import Callable

import torch
from numpy import load as np_load
from torch import as_tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class SmarTable(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.root = self.root.joinpath("SmarTable")
        if not self._check_exists(self.root):
            raise RuntimeError("Dataset folder not found!")

        self.train = train
        self.classes = [
            "knock",
            "swipe-down",
            "swipe-left",
            "swipe-right",
            "swipe-up",
            "tap",
        ]
        self.data, self.targets = (
            self._load_train_data() if train else self._load_test_data()
        )
        self.targets = list(map(lambda t: self.classes.index(t), self.targets))
        self.transform = transform
        self.target_transform = target_transform

        logger.info(
            f"Dataset: {self.__class__.__name__} ({'Training' if self.train else 'Test'})"
        )
        logger.debug(f"Total: {len(self.data)} samples")
        logger.debug(f"Classes: {self.classes}")
        logger.debug(
            f"Element Shape: {self.data[0].shape}={self.data[0].numel()} values total"
        )

    def _check_exists(self, root: Path) -> bool:
        if not root.exists():
            return False
        if not root.is_dir():
            return False
        return True

    def _load_data(self) -> tuple[list, list[str]]:
        if self.train:
            return self._load_train_data()
        else:
            return self._load_test_data()

    def _load_train_data(self) -> tuple[list, list[str]]:
        available_test_subjects = sorted(
            subdir for subdir in self.root.iterdir() if subdir.is_dir()
        )
        available_test_subjects = available_test_subjects[
            : int(0.8 * len(available_test_subjects))
        ]
        logger.debug(f"Available Subjects: {available_test_subjects}")

        if len(available_test_subjects) == 0:
            raise RuntimeError("Dataset can't be loaded!")

        data, targets = [], []
        for subject_dir in available_test_subjects:
            for session_dir in [
                session_dir
                for session_dir in subject_dir.iterdir()
                if session_dir.is_dir()
            ]:
                for class_dir in [
                    class_dir
                    for class_dir in session_dir.iterdir()
                    if class_dir.is_dir()
                ]:
                    for sample in [
                        sample for sample in class_dir.iterdir() if sample.is_file()
                    ]:
                        logger.debug(f"Loading: {sample.absolute()}")
                        with np_load(sample.absolute()) as sample_data:
                            data.append(as_tensor(sample_data["x"]))
                        targets.append(sample.parent.name)

        logger.debug(f"Loaded {len(data)} samples")
        return data, targets

    def _load_test_data(self) -> tuple[list, list[str]]:
        available_test_subjects = sorted(
            subdir for subdir in self.root.iterdir() if subdir.is_dir()
        )
        available_test_subjects = available_test_subjects[
            int(0.8 * len(available_test_subjects)) :
        ]
        logger.debug(f"Available Subjects: {available_test_subjects}")

        if len(available_test_subjects) == 0:
            raise RuntimeError("Dataset can't be loaded!")

        data, targets = [], []
        for subject_dir in available_test_subjects:
            for session_dir in [
                session_dir
                for session_dir in subject_dir.iterdir()
                if session_dir.is_dir()
            ]:
                for class_dir in [
                    class_dir
                    for class_dir in session_dir.iterdir()
                    if class_dir.is_dir()
                ]:
                    for sample in [
                        sample for sample in class_dir.iterdir() if sample.is_file()
                    ]:
                        logger.debug(f"Loading: {sample.absolute()}")
                        with np_load(sample.absolute()) as sample_data:
                            data.append(as_tensor(sample_data["x"]))
                        targets.append(sample.parent.name)

        logger.debug(f"Loaded {len(data)} samples")
        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def smartable_trainingset_flattened() -> SmarTable:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    return SmarTable(root="datasets", train=True, transform=transformations)


def smartable_validationset_flattened() -> SmarTable:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    return SmarTable(root="datasets", train=False, transform=transformations)


def smartable_dataloader(
    dataset: SmarTable, batch_size: int = 512, shuffle: bool = False
) -> DataLoader:
    logger.info(
        f"Dataloader from {dataset.__class__.__name__} with {batch_size} samples per batch ({shuffle=})"
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )
    ds = smartable_trainingset_flattened()
    dl = smartable_dataloader(ds)
    element = next(iter(dl))
    print(f"data={element[0][0]} ({element[0][0].shape})\ntarget={element[1][0]}")
