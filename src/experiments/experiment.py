import abc
from pathlib import Path

from src.config import Configuration
from src.nn.data import Dataset
from src.nn.model import Sequential


class Experiment(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "run") and callable(subclass.run)

    def __init__(self, log_dir: Path, config: Configuration) -> None: ...

    @abc.abstractmethod
    def run(self, model: Sequential, dataset: Dataset) -> None: ...
