import re

import pytest
import torch

from src.nn.training import train as training_module


class _DummyDataset:
    def __init__(self):
        self.name = "Dummy"
        self.classes = {"a": 0}
        self.training_loader = []
        self.validation_loader = []


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


def test_training_builder_requires_model():
    builder = training_module.TrainingBuilder().dataset(_DummyDataset())
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
        def __init__(self, model, dataset, loss_fn, optimiezer, device, log_dir):
            captured["model"] = model
            captured["dataset"] = dataset
            captured["loss_fn"] = loss_fn
            captured["optimizer"] = optimiezer
            captured["device"] = device
            captured["log_dir"] = log_dir

    monkeypatch.setattr(training_module, "Training", DummyTraining)

    model = torch.nn.Linear(2, 2)
    dataset = _DummyDataset()

    builder = training_module.TrainingBuilder().model(model).dataset(dataset)
    training = builder.build()

    assert isinstance(training, DummyTraining)
    assert captured["model"] is model
    assert captured["dataset"] is dataset
    assert isinstance(captured["loss_fn"], torch.nn.CrossEntropyLoss)
    assert isinstance(captured["optimizer"], torch.optim.Adam)
    assert captured["device"] == "cpu"
    assert re.match(r"^logs\/\d+$", captured["log_dir"])


def test_training_train_returns_metrics(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2)
    dataset = _DummyDataset()
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(
        training_module, "SummaryWriter", lambda log_dir: _DummyWriter()
    )

    training = training_module.Training(
        model=model,
        dataset=dataset,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimiezer=torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        log_dir=str(log_dir),
    )

    calls = {"train": [], "validate": [], "save": []}

    def fake_train_epoch(epoch):
        calls["train"].append(epoch)
        return _DummyMetrics({"loss": epoch})

    def fake_validate_epoch():
        calls["validate"].append(True)
        return _DummyMetrics({"loss": 0})

    def fake_log_metrics(epoch, training_metrics, validation_metrics):
        return None

    def fake_save_model(store_only_last_model, epoch):
        calls["save"].append((store_only_last_model, epoch))

    monkeypatch.setattr(training, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(training, "_validate_epoch", fake_validate_epoch)
    monkeypatch.setattr(training, "_log_metrics", fake_log_metrics)
    monkeypatch.setattr(training, "_save_model", fake_save_model)

    metrics = training.train(epochs=2, store_only_last_model=True)

    assert metrics == {
        1: {"training": {"loss": 0}, "validation": {"loss": 0}},
        2: {"training": {"loss": 1}, "validation": {"loss": 0}},
    }
    assert calls["train"] == [0, 1]
    assert len(calls["validate"]) == 2
    assert calls["save"] == [(True, 0), (True, 1)]


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
        dataset=dataset,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimiezer=torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
        log_dir=str(log_dir),
    )

    training._save_model(store_only_last_model=True, epoch=3)
    training._save_model(store_only_last_model=False, epoch=3)

    assert saved_paths == [
        f"{log_dir}/models/model.pth",
        f"{log_dir}/models/3.pth",
    ]
