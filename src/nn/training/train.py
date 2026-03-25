import logging
from pathlib import Path
from typing import Self

from torch import load, no_grad, optim, save
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from src.nn.data import Dataset
from src.nn.model import Sequential

from .metrics import Metrics, MetricWriter

logger = logging.getLogger(__name__)


class Training:
    def __init__(
        self,
        model: Sequential,
        dataset: Dataset,
        loss_fn: _Loss,
        optimiezer: optim.Optimizer,
        device: str,
        log_dir: Path,
        early_stopping: tuple[int, float] | None = None,
    ):
        log_dir.mkdir(exist_ok=True, parents=True)
        log_dir.joinpath("models").mkdir(exist_ok=True)
        log_dir.joinpath("models", "snapshots").mkdir(exist_ok=True)
        log_dir.joinpath("tensorboard").mkdir(exist_ok=True)

        if early_stopping is not None:
            self.early_stopping = True
            self.early_stopping_patience: int = early_stopping[0]
            self.early_stopping_threshold: float = early_stopping[1]
            self.early_stopping_counter: int = 0
            self.best_loss: float = float("inf")
        else:
            self.early_stopping = False

        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimiezer
        self.device = device
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=log_dir.joinpath("tensorboard"))

    def _check_previous_state(self, epochs: int) -> int:
        model_dir: Path = self.log_dir.joinpath("models")
        snapshot_dir: Path = model_dir.joinpath("snapshots")

        first_epoch: int

        logger.debug(f"Checking previous state: {model_dir=}, {snapshot_dir=}")

        if not model_dir.is_dir():
            logger.debug("Checking previous state: No model dir!")
            first_epoch = 0
        elif model_dir.joinpath("model.pth").exists():
            logger.debug("Checking previous state: model.pth exists!")
            first_epoch = epochs
        else:
            logger.debug("Checking previous state: Checking snapshots!")
            snapshots: list[Path] = [
                file for file in snapshot_dir.iterdir() if file.is_file()
            ]
            if not snapshots:
                logger.debug("Checking previous state: No snapshots!")
                first_epoch = 0
            else:
                logger.debug("Checking previous state: Found snapshots!")
                snapshot_epochs: list[int] = [
                    int(file.name.split("_")[-1].split(".")[0]) for file in snapshots
                ]
                first_epoch = max(snapshot_epochs) + 1
        return first_epoch

    def _train_epoch(self, epoch: int) -> Metrics:
        training_metrics = Metrics(
            loss_fn=self.loss_fn, num_classes=len(self.dataset.classes)
        )
        training_metrics.reset()

        for inputs, labels in tqdm(
            self.dataset.training_loader,
            total=len(self.dataset.training_loader),
            desc=f"Training Epoch {epoch + 1} -> Progress",
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()
            training_metrics.add(labels=labels, outputs=outputs)

        logger.info("Training Epoch Done")
        return training_metrics

    def _validate_epoch(self) -> Metrics:
        validation_metrics = Metrics(
            loss_fn=self.loss_fn, num_classes=len(self.dataset.classes)
        )
        validation_metrics.reset()

        with no_grad():
            for inputs, labels in self.dataset.validation_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                validation_metrics.add(labels=labels, outputs=outputs)

        logger.info("Validation Epoch Done")
        return validation_metrics

    def _log_metrics(
        self,
        epoch: int,
        training_metrics: Metrics,
        validation_metrics: Metrics,
    ) -> None:
        self.tb_writer.add_scalars(
            "Loss",
            {
                "Training": training_metrics.get_loss(),
                "Validation": validation_metrics.get_loss(),
            },
            epoch,
        )
        self.tb_writer.add_scalars(
            "Accuracy",
            {
                "Training": training_metrics.get_accuracy(),
                "Validation": validation_metrics.get_accuracy(),
            },
            epoch,
        )
        self.tb_writer.flush()
        logger.info("Logging Tenbsorboard Done")

        metric_path = self.log_dir.joinpath("metrics")

        training_metrics_directory = metric_path.joinpath("training")
        training_metrics_directory.mkdir(exist_ok=True, parents=True)
        training_metrics_writer = MetricWriter(training_metrics_directory)
        training_metrics_writer.write(training_metrics, False, f"epoch{epoch}.json")

        validation_metrics_directory = metric_path.joinpath("validation")
        validation_metrics_directory.mkdir(exist_ok=True, parents=True)
        validation_metrics_writer = MetricWriter(validation_metrics_directory)
        validation_metrics_writer.write(validation_metrics, False, f"epoch{epoch}.json")
        logger.info("Logging Metrics Done")

    def _save_snapshot(self, store_only_last_model: bool, epoch: int) -> None:
        if store_only_last_model:
            for file in self.log_dir.joinpath("models", "snapshots").iterdir():
                if file.is_file():
                    file.unlink()
        save(
            self.model.state_dict(),
            self.log_dir.joinpath("models", "snapshots", f"snapshot_{epoch}.pth"),
        )

    def _check_early_stopping(self, epoch: int, loss: float) -> bool:
        if not self.early_stopping:
            return False

        if (self.best_loss - loss) > self.early_stopping_threshold:
            logger.debug(f"reset early stopping counter ({loss=})")
            self.best_loss = loss
            self.early_stopping_counter = 0
        else:
            logger.debug("increasing early stopping counter")
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            return True
        return False

    def _clean_snapshots(self) -> None:
        if self.log_dir.joinpath("models", "snapshots").is_dir():
            for file in self.log_dir.joinpath("models", "snapshots").iterdir():
                if file.is_file():
                    file.unlink()
            self.log_dir.joinpath("models", "snapshots").rmdir()

    def _save_model(self) -> None:
        model_path: Path = self.log_dir.joinpath("models", "model.pth")
        save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def train(
        self, epochs: int = 100, store_only_last_model: bool = False, skip: bool = True
    ) -> dict[int, dict]:
        metrics: dict[int, dict] = {}
        self.model.to(self.device)

        if skip:
            begin: int = self._check_previous_state(epochs)
            if begin == epochs:
                logger.info("Skipping all epochs!")
                self.model.load_state_dict(
                    load(self.log_dir.joinpath("models", "model.pth"))
                )
            elif begin != 0:
                logger.info(f"Skipping {begin} epochs!")
                self.model.load_state_dict(
                    load(
                        self.log_dir.joinpath(
                            "models", "snapshots", f"snapshot_{begin - 1}.pth"
                        )
                    )
                )
        else:
            begin = 0

        for epoch in range(begin, epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            self.model.train(True)
            training_metrics = self._train_epoch(epoch)
            validation_metrics = self._validate_epoch()
            self._log_metrics(epoch, training_metrics, validation_metrics)
            self._save_snapshot(store_only_last_model, epoch)

            metrics[epoch + 1] = {
                "training": training_metrics.get(),
                "validation": validation_metrics.get(),
            }

            if self._check_early_stopping(epoch, validation_metrics.get_loss()):
                logger.info("Early Stopping Triggered!")
                break

        self._save_model()
        if store_only_last_model:
            self._clean_snapshots()
        logger.debug("Training Done")
        return metrics


class TrainingBuilder:
    def __init__(self):
        self._model: Sequential | None = None
        self._dataset: Dataset | None = None
        self._loss_fn: _Loss | None = None
        self._optimizer: optim.Optimizer | None = None
        self._device: str | None = None
        self._log_dir: Path | None = None
        self._early_stopping: tuple[int, float] | None = None

    def model(self, model: Sequential) -> Self:
        self._model = model
        logger.debug(f"add model: {model}")
        return self

    def dataset(self, dataset: Dataset) -> Self:
        self._dataset = dataset
        logger.debug(f"add dataset: {dataset}")
        return self

    def loss_fn(self, loss_fn: _Loss) -> Self:
        self._loss_fn = loss_fn
        logger.debug(f"add loss_fn: {loss_fn}")
        return self

    def optimizer(self, optimizer: optim.Optimizer) -> Self:
        self._optimizer = optimizer
        logger.debug(f"add optimizer: {optimizer}")
        return self

    def device(self, device: str) -> Self:
        self._device = device
        logger.debug(f"add device: {device}")
        return self

    def log_dir(self, log_dir: Path) -> Self:
        self._log_dir = log_dir
        logger.debug(f"add tensorboard_log_dir: {log_dir}")
        return self

    def early_stopping(self, patience: int, threshold: float) -> Self:
        self._early_stopping = (patience, threshold)
        logger.debug(f"add early_stopping: ({patience=}, {threshold=})")
        return self

    def build(self) -> Training:
        """Build Training instance.

        Raises:
            VALUE_ERROR: If required fields are missing.

        Returns:
            Training: Training instance.

        """
        if self._model is None:
            raise ValueError("TrainingBuilder missing required Model!")
        if self._dataset is None:
            raise ValueError("TrainingBuilder missing required Dataset!")

        loss_fn = self._loss_fn or CrossEntropyLoss()
        optimizer = self._optimizer or optim.Adam(self._model.parameters())
        device = self._device or "cpu"
        log_dir = self._log_dir or Path("logs/{int(datetime.now().timestamp() * 1000)}")

        logger.debug(
            f"build Training: {self._model=}, {self._dataset=}, {loss_fn=}, {optimizer=}, {device=}, {log_dir=}, {self._early_stopping=}"
        )

        return Training(
            model=self._model,
            dataset=self._dataset,
            loss_fn=loss_fn,
            optimiezer=optimizer,
            device=device,
            log_dir=log_dir,
            early_stopping=self._early_stopping,
        )
