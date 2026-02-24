import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.modules.loss import _Loss


class Metrics:
    _processed_samples: int
    _processed_batches: int
    _running_loss: float
    _correct_samples: int
    _confusion_matrix: torch.Tensor

    _num_classes: int
    _loss_fn: _Loss

    def __init__(self, loss_fn: _Loss, num_classes: int) -> None:
        self._loss_fn = loss_fn
        self._num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._processed_batches = 0
        self._processed_samples = 0
        self._correct_samples = 0
        self._running_loss = 0.0
        self._confusion_matrix = torch.zeros(
            (self._num_classes, self._num_classes), dtype=torch.int64
        )

    def _extend_loss(self, labels: torch.Tensor, outputs: torch.Tensor) -> None:
        self._running_loss += self._loss_fn(outputs, labels).item()

    def _extend_confusion_matrix(
        self, labels: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        for target, prediction in zip(labels, outputs.argmax(dim=1)):
            self._confusion_matrix[target, prediction] += 1

    def add(self, labels: torch.Tensor, outputs: torch.Tensor) -> None:
        self._processed_batches += 1
        self._processed_samples += len(labels)
        self._correct_samples += int((outputs.argmax(dim=1) == labels).sum().item())

        self._extend_loss(labels=labels, outputs=outputs)
        self._extend_confusion_matrix(labels=labels, outputs=outputs)

    def get_loss(self) -> float:
        """Get The loss based on current recordings.

        Loss is calculated  based on the provided loss function.

        Returns:
            loss (float): current loss value
        """
        return self._running_loss / self._processed_batches

    def get_accuracy(self) -> float:
        """Get the accuracy based on current recordings.

        Accuracy is calculated as number of correct samples divided by number of
        total samples.

        Returns:
            accuracy (float): current accuracy value
        """
        return float(float(self._correct_samples) / float(self._processed_samples))

    def _precision(self) -> torch.Tensor:
        true_positives = torch.diag(self._confusion_matrix).float()
        false_positives = self._confusion_matrix.sum(dim=0).float() - true_positives
        precision_per_class = true_positives / (true_positives + false_positives)
        return precision_per_class

    def get_precision(self, class_id: int) -> float:
        """Get the precision for the given class.

        The precision is calculated as the number of true positives divided by the
        number of true positives and false positives.

        Args:
            class_id (int): class id to calculate precision for

        Returns:
            precision (float): current precision value of given class
        """
        self._check_class_id_in_range(class_id)
        return float(self._precision()[class_id])

    def _recall(self) -> torch.Tensor:
        true_positives = torch.diag(self._confusion_matrix).float()
        false_negatives = self._confusion_matrix.sum(dim=1).float() - true_positives
        recall_per_class = true_positives / (true_positives + false_negatives)
        return recall_per_class

    def get_recall(self, class_id: int) -> float:
        """Get the recall for the given class.

        The recall is calculated as the number of true positives divided by the
        number of true positives and false negatives.

        Args:
            class_id (int): class id to calculate recall for

        Returns:
            recall (float): current recall value of given class
        """
        self._check_class_id_in_range(class_id)
        return float(self._recall()[class_id])

    def _f1(self) -> torch.Tensor:
        precision_per_class = self._precision()
        recall_per_class = self._recall()

        f1_per_class = (2 * precision_per_class * recall_per_class) / (
            precision_per_class + recall_per_class
        )
        return f1_per_class

    def get_f1(self, class_id: int) -> float:
        """Get the F1 for the given class.

        The F1 is calculated as the harmonic mean of precision and recall.

        Args:
            class_id (int): class id to calculate F1 for

        Returns:
            f1 (float): current F1 value of given class
        """
        self._check_class_id_in_range(class_id)
        return float(self._f1()[class_id])

    def get(
        self, include_meta: bool = False
    ) -> (
        dict[str, float | list[float]] | dict[str, float | list[float] | dict[str, Any]]
    ):
        """Return the metrics as a dictionary.

        Args:
            include_meta (bool): whether to include meta data in the returned dictionary

        Returns:
            metrics (dict): dictionary containing the metrics
        """
        metrics: dict = {}
        metrics["loss"] = self.get_loss()
        metrics["accuracy"] = self.get_accuracy()
        metrics["precision"] = self._precision().tolist()
        metrics["recall"] = self._recall().tolist()
        metrics["f1"] = self._f1().tolist()
        if include_meta:
            metrics["meta_data"] = {}
            metrics["meta_data"]["num_classes"] = self._num_classes
            metrics["meta_data"]["loss_fn"] = self._loss_fn.__class__.__name__
            metrics["meta_data"]["processed_batches"] = self._processed_batches
            metrics["meta_data"]["processed_samples"] = self._processed_samples
            metrics["meta_data"]["correct_samples"] = self._correct_samples
            metrics["meta_data"]["confusion_matrix"] = self._confusion_matrix.tolist()
        return metrics

    def _check_class_id_in_range(self, class_id: int) -> None:
        if class_id < 0 or class_id >= self._num_classes:
            raise ValueError(f"class_id {class_id} is out of range")


class MetricWriter:
    _storage_directory: Path

    def __init__(self, directory: Path) -> None:
        """Create a MetricWriter instance.

        Args:
            directory (str): directory to write metrics to

        Raises:
            FileNotFoundError: if the parent directory does not exist
            NotADirectoryError: if the parent directory is not a directory
            PermissionError: if the parent directory is not writable
            OSError: if the parent directory cannot be created

        Returns:
            None
        """
        self._storage_directory = directory

        if self._storage_directory is None:
            raise ValueError("storage_directory cannot be None")

        try:
            self._storage_directory.mkdir(exist_ok=True, parents=True)
        except (FileNotFoundError, NotADirectoryError):
            raise ValueError("parent directories must exist")
        except PermissionError:
            raise RuntimeError("parent directories must be writable")
        except OSError:
            raise RuntimeError("cannot create directory")

    def write(
        self, metrics: Metrics, write_meta: bool = True, filename: str | None = None
    ) -> None:
        metrics_dict = metrics.get(include_meta=write_meta)
        with open(
            self._storage_directory.joinpath(
                filename or f"metrics_{datetime.now(timezone.utc).timestamp()}.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics_dict, f, indent=4)
