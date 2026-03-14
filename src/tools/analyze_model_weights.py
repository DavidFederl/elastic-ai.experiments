from pathlib import Path
from typing import Annotated

import torch
import typer

from src.nn.model.serialize import load_from_json


def main(
    source: Annotated[
        Path,
        typer.Argument(
            help="Configuration file defining the experiment.",
            exists=True,
            file_okay=True,
            writable=True,
            readable=True,
        ),
    ],
):
    data: dict[str, torch.Tensor] = load_from_json(source)
    for name, tensor in data.items():
        print(f"{name}: min={tensor.min().item()}; max={tensor.max().item()}")

    combined_data = torch.cat(list(map(lambda t: t.flatten(), data.values())))
    print(
        f"TOTAL: min={combined_data.min().item()} ({combined_data.argmin()}); max={combined_data.max()} ({combined_data.argmax()})"
    )


if __name__ == "__main__":
    typer.run(main)
