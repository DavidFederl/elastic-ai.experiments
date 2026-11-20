import logging
from pathlib import Path
from typing import Callable

from numpy import load as np_load
from torch import Tensor, from_numpy
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class Quiroga2020(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.root = self.root.joinpath("Quiroga2020")
        if not self._check_exists(self.root):
            raise RuntimeError("Dataset folder not found!")

        self.train = train
        self.classes = [f"unit_{i}" for i in range(6)]
        self.data, self.targets = (
            self._load_train_data() if train else self._load_test_data()
        )
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

    def _find_data_file(self) -> Path:
        matches = sorted(self.root.glob("*.npy"))
        if len(matches) == 0:
            raise RuntimeError("Dataset file not found!")
        return matches[0]

    def _load_data(self) -> tuple[Tensor, list[int]]:
        raw = np_load(self._find_data_file(), allow_pickle=True).item()
        data = from_numpy(raw["data"]).float()
        targets = raw["label"].tolist()
        return data, targets

    def _load_train_data(self) -> tuple[Tensor, list[int]]:
        data, targets = self._load_data()
        split_idx = int(0.8 * len(data))
        logger.debug(f"Loaded {split_idx} samples")
        return data[:split_idx], targets[:split_idx]

    def _load_test_data(self) -> tuple[Tensor, list[int]]:
        data, targets = self._load_data()
        split_idx = int(0.8 * len(data))
        logger.debug(f"Loaded {len(data) - split_idx} samples")
        return data[split_idx:], targets[split_idx:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def quiroga2020_trainingset_flattened() -> Quiroga2020:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.Lambda(lambda x: Tensor.flatten(x)),
        ]
    )
    return Quiroga2020(root="datasets", train=True, transform=transformations)


def quiroga2020_validationset_flattened() -> Quiroga2020:
    transformations: transforms.Compose = transforms.Compose(
        [
            transforms.Lambda(lambda x: Tensor.flatten(x)),
        ]
    )
    return Quiroga2020(root="datasets", train=False, transform=transformations)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )
    ds = quiroga2020_trainingset_flattened()
    print(f"data={ds[0][0]} ({ds[0][0].shape})\ntarget={ds[1][0]}")
