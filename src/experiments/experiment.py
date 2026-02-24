from abc import ABC, abstractmethod
from pathlib import Path

from src.nn.data import Dataset
from src.nn.model import Sequential


class Experiment(ABC):
    log_dir: Path

    @abstractmethod
    def __init__(self, log_dir: Path, **kwargs) -> None: ...

    @abstractmethod
    def run(self, model: Sequential, dataset: Dataset) -> None: ...
