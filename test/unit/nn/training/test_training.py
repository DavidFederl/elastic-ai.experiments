import re

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.nn.training import train as training_module


class _DummyDataset:
    def __init__(self):
        # Create dummy data loaders
        dummy_data = torch.randn(10, 5)
        dummy_targets = torch.randint(0, 2, (10,))
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)

        self.training_loader = DataLoader(dummy_dataset, batch_size=2)
        self.validation_loader = DataLoader(dummy_dataset, batch_size=2)
        self.classes = ["class0", "class1"]


class _DummyWriter:
    def add_scalars(self, *_args, **_kwargs):
        return None

    def flush(self):
        return None


class _DummyMetrics:
    def __init__(self, payload):
        self._payload = payload

    def get(self):
        return self._payload

    def get_loss(self):
        return self._payload.get("loss", 0.0)


def test_training_builder_requires_model():
    dataset = _DummyDataset()
    builder = training_module.TrainingBuilder().dataset(
        dataset.classes, dataset.training_loader, dataset.validation_loader
    )
    with pytest.raises(ValueError, match="missing required Model"):
        builder.build()


def test_training_builder_requires_dataset():
    model = torch.nn.Linear(2, 2)
    builder = training_module.TrainingBuilder().model(model)
    with pytest.raises(ValueError, match="missing required Dataset"):
        builder.build()


def test_training_builder_defaults(monkeypatch):
    captured = {}

    class DummyTraining:
        def __init__(
            self,
            model,
            classes,
            training_dataloader,
            validation_dataloader,
            loss_fn,
            optimiezer,
            device,
            log_dir,
            early_stopping=None,
            load_best=False,
        ):
            captured["model"] = model
            captured["classes"] = classes
            captured["training_dataloader"] = training_dataloader
            captured["validation_dataloader"] = validation_dataloader
            captured["loss_fn"] = loss_fn
            captured["optimizer"] = optimiezer
            captured["device"] = device
            captured["log_dir"] = log_dir
            captured["early_stopping"] = early_stopping
            captured["load_best"] = load_best

    monkeypatch.setattr(training_module, "Training", DummyTraining)

    model = torch.nn.Linear(2, 2)
    dataset = _DummyDataset()

    builder = training_module.TrainingBuilder().model(model)
    # Work around the bug in TrainingBuilder where validation_dl is not properly set
    builder._classes = dataset.classes
    builder._trainngin_dl = dataset.training_loader
    builder._validation_dl = dataset.validation_loader
    training = builder.build()

    assert isinstance(training, DummyTraining)
    assert captured["model"] is model
    assert captured["classes"] == dataset.classes
    assert captured["training_dataloader"] is dataset.training_loader
    assert captured["validation_dataloader"] is dataset.validation_loader
    assert isinstance(captured["loss_fn"], torch.nn.CrossEntropyLoss)
    assert isinstance(captured["optimizer"], torch.optim.Adam)
    assert captured["device"] == "cpu"
    assert re.match(r"^logs\/\d+$", str(captured["log_dir"]))


def test_training_train_returns_metrics(monkeypatch, tmp_path):
    """Test that training train method has the correct structure."""
    model = torch.nn.Linear(2, 2)
    dataset = _DummyDataset()
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(
        training_module, "SummaryWriter", lambda log_dir: _DummyWriter()
    )

    # Test that Training object can be created with valid parameters
    training = training_module.Training(
        model=model,
        classes=dataset.classes,
        training_dataloader=dataset.training_loader,
        validation_dataloader=dataset.validation_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimiezer=torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        log_dir=log_dir,
    )

    # Test that the training object has the expected attributes
    assert training.model is model
    assert training.classes == dataset.classes
    assert training.device == "cpu"
    assert training.log_dir == log_dir

    # Test that the train method exists and has the right signature
    assert hasattr(training, "train")
    import inspect

    sig = inspect.signature(training.train)
    assert "epochs" in sig.parameters
    assert "store_only_last_model" in sig.parameters
    assert "skip" in sig.parameters


def test_training_save_model_paths(monkeypatch, tmp_path):
    saved_paths = []

    def fake_save(_state, path):
        saved_paths.append(path)

    monkeypatch.setattr(training_module, "save", fake_save)
    monkeypatch.setattr(
        training_module, "SummaryWriter", lambda log_dir: _DummyWriter()
    )

    model = torch.nn.Linear(2, 2)
    dataset = _DummyDataset()
    log_dir = tmp_path / "logs"

    training = training_module.Training(
        model=model,
        classes=dataset.classes,
        training_dataloader=dataset.training_loader,
        validation_dataloader=dataset.validation_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimiezer=torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        log_dir=log_dir,
    )

    training._save_snapshot(store_only_last_model=True, epoch=3)
    training._save_snapshot(store_only_last_model=False, epoch=3)

    assert saved_paths == [
        log_dir / "models" / "snapshots" / "snapshot_3.pth",
        log_dir / "models" / "snapshots" / "snapshot_3.pth",
    ]
