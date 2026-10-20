"""Convert a torchvision-style dataset class into a single HDF5 file.

Writes two files into the storage path:

    <name>.h5   — groups `train` and `test`, each with datasets `data` and `targets`
    <name>.txt  — human-readable description of the file layout

The dataset class is given as `module:ClassName` (e.g. `src.nn.data.cifar10:CIFAR10`)
and must accept `root`, `train` and `transform` keyword arguments. It is instantiated
without transform, so samples are stored raw (e.g. uint8 images) and normalization /
augmentation stays a training-time concern.

Example:
    uv run src/tools/dataset_to_h5.py src.nn.data.cifar10:CIFAR10 /path/to/storage
"""

import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer()
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
_SPLITS = {"train": True, "test": False}


def load_dataset_class(spec: str) -> type:
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise typer.BadParameter("expected format `module:ClassName`")
    sys.path.insert(0, str(_REPO_ROOT))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def read_split(dataset) -> tuple[np.ndarray, np.ndarray]:
    samples, targets = [], []
    for sample, target in tqdm(dataset, desc="Reading", leave=False):
        samples.append(np.asarray(sample))
        targets.append(int(target))
    return np.stack(samples), np.asarray(targets, dtype=np.int64)


def describe(
    name: str,
    dataset_spec: str,
    classes: list[str] | None,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> str:
    lines = [
        f"Dataset: {name}",
        f"Source class: {dataset_spec}",
        f"Created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"File: {name}.h5 (HDF5, gzip-compressed)",
        "",
        "Layout:",
    ]
    for split, (data, targets) in splits.items():
        lines += [
            f"  /{split}/data     shape={data.shape} dtype={data.dtype}  (samples as stored by the source dataset, no transforms applied)",
            f"  /{split}/targets  shape={targets.shape} dtype={targets.dtype}  (class index per sample)",
        ]
    if classes:
        lines += ["", "Classes (index: name):"]
        lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    lines += [
        "",
        "Usage:",
        "  with h5py.File('<name>.h5', 'r') as f:",
        "      x, y = f['train/data'][i], f['train/targets'][i]",
    ]
    return "\n".join(lines) + "\n"


@app.command()
def convert(
    dataset: Annotated[str, typer.Argument(help="Dataset class as `module:ClassName`")],
    storage_path: Annotated[
        Path, typer.Argument(help="Folder receiving the .h5 and .txt files")
    ],
    root: Annotated[
        Path, typer.Option(help="Root passed to the dataset (download/cache location)")
    ] = Path("datasets"),
    name: Annotated[
        str | None,
        typer.Option(help="Output file stem (default: lowercase class name)"),
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    dataset_class = load_dataset_class(dataset)
    name = name or dataset_class.__name__.lower()
    storage_path.mkdir(parents=True, exist_ok=True)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    classes: list[str] | None = None
    for split, is_train in _SPLITS.items():
        ds = dataset_class(root=root, train=is_train, transform=None)
        classes = classes or getattr(ds, "classes", None)
        splits[split] = read_split(ds)

    h5_path = storage_path / f"{name}.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["source_class"] = dataset
        for split, (data, targets) in splits.items():
            group = f.create_group(split)
            group.create_dataset(
                "data", data=data, compression="gzip", chunks=(1, *data.shape[1:])
            )
            group.create_dataset("targets", data=targets)
        if classes:
            f.attrs["classes"] = classes

    txt_path = storage_path / f"{name}.txt"
    txt_path.write_text(describe(name, dataset, classes, splits))
    logger.info(f"Wrote {h5_path} and {txt_path}")


if __name__ == "__main__":
    app()
