import json

import pytest
import torch

from src.nn.training.metrics import Metrics, MetricWriter


def test_metrics_accumulates_loss_and_accuracy():
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = Metrics(loss_fn=loss_fn, num_classes=2)

    labels = torch.tensor([0, 1, 0, 1])
    outputs = torch.tensor(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ]
    )

    metrics.add(labels=labels, outputs=outputs)

    expected_loss = loss_fn(outputs, labels).item()
    assert metrics.get_loss() == pytest.approx(expected_loss)
    assert metrics.get_accuracy() == pytest.approx(0.5)


def test_metrics_precision_recall_f1_and_meta():
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = Metrics(loss_fn=loss_fn, num_classes=2)

    labels = torch.tensor([0, 1, 0, 1])
    outputs = torch.tensor(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ]
    )

    metrics.add(labels=labels, outputs=outputs)

    assert metrics.get_precision(0) == pytest.approx(0.5)
    assert metrics.get_recall(0) == pytest.approx(0.5)
    assert metrics.get_f1(0) == pytest.approx(0.5)
    assert metrics.get_precision(1) == pytest.approx(0.5)
    assert metrics.get_recall(1) == pytest.approx(0.5)
    assert metrics.get_f1(1) == pytest.approx(0.5)

    metrics_dict = metrics.get(include_meta=True)
    assert metrics_dict["meta_data"]["num_classes"] == 2
    assert metrics_dict["meta_data"]["processed_batches"] == 1
    assert metrics_dict["meta_data"]["processed_samples"] == 4
    assert metrics_dict["meta_data"]["correct_samples"] == 2
    assert metrics_dict["meta_data"]["confusion_matrix"] == [[1, 1], [1, 1]]


def test_metrics_rejects_out_of_range_class():
    metrics = Metrics(loss_fn=torch.nn.CrossEntropyLoss(), num_classes=2)
    with pytest.raises(ValueError, match="class_id 2 is out of range"):
        metrics.get_precision(2)


def test_metric_writer_writes_json(tmp_path):
    metrics = Metrics(loss_fn=torch.nn.CrossEntropyLoss(), num_classes=2)
    labels = torch.tensor([0, 1])
    outputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metrics.add(labels=labels, outputs=outputs)

    writer = MetricWriter(str(tmp_path))
    writer.write(metrics, write_meta=False, filename="metrics.json")

    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "meta_data" not in data
    assert data["accuracy"] == pytest.approx(1.0)


def test_metric_writer_validates_directory(tmp_path):
    with pytest.raises(ValueError, match="storage_directory cannot be None"):
        MetricWriter(None)

    missing_dir = tmp_path / "missing" / "child"
    with pytest.raises(ValueError, match="parent directories must exist"):
        MetricWriter(str(missing_dir))
